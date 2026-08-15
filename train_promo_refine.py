# -*- coding: utf-8 -*-
"""
train_promo_refine.py

手動クロップ実績から、promoモードのクロップを微調整するモデルを学習する。

promoモードは固定ルールで既に手動クロップとよく一致するが、半身写真や複数人
写真では写真ごとの判断が残る。本スクリプトは「promoの結果と手動クロップのズレ」
を学習し、その差分を予測できるようにする(詳細は promo_refine.py の説明を参照)。

使い方:
    python train_promo_refine.py --crop-csv hand_crop.csv

    # 学習だけでなく、交差検証で改善幅も確認する(推奨)
    python train_promo_refine.py --crop-csv hand_crop.csv --cross-validate

依存: scikit-learn

【交差検証について】
連番ブロック単位で分割する。連続する写真は同じ衣装・セットアップでほぼ同一構図に
なるため、ランダム分割では学習側と評価側に実質同じ写真が入り、改善幅を過大に
見積もってしまう。
"""

import argparse
import pickle
import statistics
from datetime import datetime, timezone

import numpy as np

import crop_calculator as cc
import promo_refine as pr
import evaluate_crop as ev

TARGETS = ("d_cx", "d_cy", "d_scale")


def build_dataset(items):
    """キャッシュした検出結果から、特徴量と目標値を作る。"""
    X, Y, base, hand, meta = [], [], [], [], []
    for it in items:
        W, H = it["W"], it["H"]
        poses = [[ev._Pt(*p) for p in ps] for ps in it["poses"]]
        info = cc.combine_person_boxes([cc.person_box(p, W, H) for p in poses], W, H)
        if info is None:
            continue
        info["W"], info["H"] = W, H
        base_crop = cc.compute_crop(info)
        target = pr.crop_target(base_crop, it["hand"], info)
        if target is None:
            continue
        feats = pr.extract_features(info, poses, it["detect_stage"])
        X.append([feats[n] for n in pr.FEATURE_NAMES])
        Y.append(target)
        base.append(base_crop)
        hand.append(it["hand"])
        meta.append({"seq": it["seq"], "info": info, "filename": it["filename"]})
    return np.array(X), np.array(Y), base, hand, meta


def _iou_of(crop_a, crop_b, W, H):
    a = (crop_a["CropLeft"] * W, crop_a["CropTop"] * H, crop_a["CropRight"] * W, crop_a["CropBottom"] * H)
    b = (crop_b["CropLeft"] * W, crop_b["CropTop"] * H, crop_b["CropRight"] * W, crop_b["CropBottom"] * H)
    return ev.iou(a, b)


def fit_models(X, Y, make_regressor):
    return {name: make_regressor().fit(X, Y[:, j]) for j, name in enumerate(TARGETS)}


def cross_validate(X, Y, base, hand, meta, make_regressor, folds, block_size):
    """ブロック単位の交差検証で、実際の改善幅を測る。"""
    blocks = np.array([m["seq"] // block_size for m in meta])
    unique = sorted(set(blocks.tolist()))
    before, after = [], []
    per_class = {"全身": ([], []), "半身": ([], []), "複数人": ([], [])}

    for k in range(folds):
        test_blocks = [b for i, b in enumerate(unique) if i % folds == k]
        test = np.array([b in test_blocks for b in blocks])
        train = ~test
        if test.sum() == 0 or train.sum() == 0:
            continue
        models = fit_models(X[train], Y[train], make_regressor)
        preds = np.column_stack([models[n].predict(X[test]) for n in TARGETS])

        for row, idx in enumerate(np.where(test)[0]):
            info = meta[idx]["info"]
            W, H = info["W"], info["H"]
            refined = pr.apply_delta(base[idx], preds[row], info)
            b = _iou_of(hand[idx], base[idx], W, H)
            a = _iou_of(hand[idx], refined, W, H)
            before.append(b)
            after.append(a)
            key = "全身" if info["full_body"] else "半身"
            per_class[key][0].append(b)
            per_class[key][1].append(a)
            if info["n_persons"] > 1:
                per_class["複数人"][0].append(b)
                per_class["複数人"][1].append(a)

    print(f"\n交差検証({folds}分割、ブロックサイズ {block_size}枚)")
    print(f"  全体      {len(before):4d}枚  {statistics.mean(before):.4f} -> "
          f"{statistics.mean(after):.4f}  ({statistics.mean(after) - statistics.mean(before):+.4f})")
    for label, (b, a) in per_class.items():
        if b:
            print(f"  {label:<8} {len(b):4d}枚  {statistics.mean(b):.4f} -> "
                  f"{statistics.mean(a):.4f}  ({statistics.mean(a) - statistics.mean(b):+.4f})")
    improved = sum(1 for b, a in zip(before, after) if a > b + 0.01)
    worsened = sum(1 for b, a in zip(before, after) if a < b - 0.01)
    print(f"  改善 {improved}枚 / 悪化 {worsened}枚")
    return statistics.mean(after) - statistics.mean(before)


def main():
    ap = argparse.ArgumentParser(description="promoモードのクロップを微調整するモデルを学習")
    ap.add_argument("--crop-csv", required=True, help="ExportCropHistory.lua が出力したCSV")
    ap.add_argument("--output-model", default="promo_refine_model.pkl")
    ap.add_argument("--cache", default="eval_cache.pkl", help="人物検出結果のキャッシュ")
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--cross-validate", action="store_true", help="交差検証で改善幅を確認する")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--block-size", type=int, default=ev.DEFAULT_BLOCK_SIZE)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    args = ap.parse_args()

    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        raise SystemExit("scikit-learn が見つかりません。`pip install scikit-learn` を実行してください。")

    def make_regressor():
        return GradientBoostingRegressor(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            learning_rate=args.learning_rate, random_state=0,
        )

    rows, rotated = ev.load_hand_crops(args.crop_csv)
    print(f"手動クロップ: {len(rows)}枚 (回転クロップ {rotated}枚は対象外)")
    items = ev.build_cache(rows, args.cache, args.refresh_cache)

    X, Y, base, hand, meta = build_dataset(items)
    print(f"学習データ: {len(X)}枚  特徴量 {X.shape[1]}次元")
    if len(X) < 50:
        raise SystemExit("学習に使えるデータが少なすぎます(50枚程度は必要)。")

    if args.cross_validate:
        cross_validate(X, Y, base, hand, meta, make_regressor, args.folds, args.block_size)

    # 最終モデルは全データで学習する
    models = fit_models(X, Y, make_regressor)
    bundle = {
        "feature_names": list(pr.FEATURE_NAMES),
        "n_samples": len(X),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    bundle.update(models)
    with open(args.output_model, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nモデルを保存しました: {args.output_model}")
    print("使用するには:")
    print(f"  python crop_calculator.py --input <フォルダ> --output crop_data.csv "
          f"--mode promo --refine-model {args.output_model}")


if __name__ == "__main__":
    main()
