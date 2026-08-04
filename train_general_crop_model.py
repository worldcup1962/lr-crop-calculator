# -*- coding: utf-8 -*-
"""
train_general_crop_model.py

Lightroomから書き出したクロップ履歴CSV(ExportCropHistory.luaの出力)と、
その元画像ファイル群から、汎用ポートレートクロップ(--mode general)用の
回帰モデルを学習する。

入力CSVの列(ExportCropHistory.lua の出力形式):
    filename, path, raw_W, raw_H, CropTop, CropLeft, CropRight, CropBottom,
    CropAngle, is_cropped

処理の流れ:
    1. is_cropped=1 かつ CropAngle が実質0の行のみ対象にする
       (本ツールは水平・垂直軸に沿ったクロップのみ扱うため、回転クロップは除外)
    2. path列(またはimages-dir内の同名ファイル)から元画像を特定
    3. crop_calculator.analyze_image() で人物のポーズ・バウンディングボックスを検出
       (crop_calculator.py と全く同じ検出ロジックを使うことで、学習時と推論時の
       特徴量の前提を一致させる)
    4. Lightroom側のクロップ値(保存時/raw座標系)を crop_calculator.raw_crop_to_display()
       で表示向き座標に変換し、人物バウンディングボックスに対する相対余白
       (L, R, T, B)を教師信号として計算する
    5. general_crop.extract_features() で特徴量を抽出する
    6. 4つの独立した GradientBoostingRegressor(L/R/T/B)を学習し、
       ホールドアウトデータでMAEと、実際のクロップ枠とのIoUを表示する

使い方:
    python train_general_crop_model.py --crop-csv crop_history.csv \
        --images-dir /path/to/jpgs --output-model general_crop_model.pkl

依存関係: scikit-learn が別途必要 (pip install scikit-learn)
"""

import argparse
import csv
import os
import pickle
from datetime import datetime, timezone

import numpy as np

import crop_calculator as cc
import general_crop as gc

CROP_ANGLE_EPS = 0.01
# 明らかにデータ異常(例: 誤ってフルフレームをクロップとして記録した等)と思われる
# 極端な余白は学習データから除外する
TARGET_CLAMP = (-0.6, 3.0)


def _truthy(s):
    return str(s).strip() in ("1", "true", "True", "TRUE")


def build_filename_index(images_dir):
    """ファイル名 -> パスの索引。同名ファイルが複数ある場合は None を格納し、
    「特定できない」ことを表す。

    Canonの連番ファイル名は撮影日フォルダをまたいで重複するため、
    適当に1つ選ぶと、クロップ値と画像の組み合わせが食い違った学習データになる。
    それを防ぐため、曖昧なものは使わない。
    """
    index = {}
    ambiguous = set()
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg")):
                if f in index:
                    ambiguous.add(f)
                    index[f] = None
                elif f not in ambiguous:
                    index[f] = os.path.join(root, f)
    if ambiguous:
        print(f"警告: images-dir内で同名ファイルが{len(ambiguous)}種類重複しています。"
              "これらはパス列で特定できない限りスキップします。")
    return index


def resolve_path(row, filename_index):
    p = (row.get("path") or "").strip()
    if p and os.path.exists(p):
        return p
    filename = (row.get("filename") or "").strip()
    if filename_index is not None:
        # 同名重複で特定できないものは None が入っている
        return filename_index.get(filename)
    return None


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def main():
    parser = argparse.ArgumentParser(description="汎用ポートレートクロップ用モデルの学習")
    parser.add_argument("--crop-csv", required=True, help="ExportCropHistory.lua の出力CSV")
    parser.add_argument("--images-dir", default=None,
                         help="元画像フォルダ(CSVのpath列が使えない場合のフォールバック検索先)")
    parser.add_argument("--output-model", default="general_crop_model.pkl")
    parser.add_argument("--test-size", type=float, default=0.15)
    args = parser.parse_args()

    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
    except ImportError:
        raise SystemExit(
            "scikit-learn が見つかりません。`pip install scikit-learn` を実行してください。"
        )

    with open(args.crop_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"CSV読み込み: {len(rows)} 行")

    filename_index = build_filename_index(args.images_dir) if args.images_dir else None

    stats = {
        "not_cropped": 0, "rotated": 0, "missing_file": 0,
        "no_person": 0, "outlier": 0, "used": 0,
    }

    features_list = []
    targets = {"L": [], "R": [], "T": [], "B": []}
    eval_records = []  # (features, targets, person_box, gt_box) すべて保持(IoU評価用)

    cc.ensure_model()
    landmarker = cc.create_landmarker()
    try:
        for i, row in enumerate(rows, 1):
            if not _truthy(row.get("is_cropped", "0")):
                stats["not_cropped"] += 1
                continue

            angle = float(row.get("CropAngle") or 0.0)
            if abs(angle) > CROP_ANGLE_EPS:
                stats["rotated"] += 1
                continue

            path = resolve_path(row, filename_index)
            if not path or not os.path.exists(path):
                stats["missing_file"] += 1
                continue

            info = cc.analyze_image(path, landmarker)
            if info is None:
                stats["no_person"] += 1
                continue

            crop_raw = {
                "CropTop": float(row["CropTop"]), "CropLeft": float(row["CropLeft"]),
                "CropRight": float(row["CropRight"]), "CropBottom": float(row["CropBottom"]),
            }
            crop_disp = cc.raw_crop_to_display(crop_raw, info["orientation"])

            W, H = info["W"], info["H"]
            x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]
            person_w, person_h = x2 - x1, y2 - y1

            gt_left = crop_disp["CropLeft"] * W
            gt_right = crop_disp["CropRight"] * W
            gt_top = crop_disp["CropTop"] * H
            gt_bottom = crop_disp["CropBottom"] * H

            L = (x1 - gt_left) / person_w
            R = (gt_right - x2) / person_w
            T = (y1 - gt_top) / person_h
            B = (gt_bottom - y2) / person_h

            lo, hi = TARGET_CLAMP
            if not all(lo <= v <= hi for v in (L, R, T, B)):
                stats["outlier"] += 1
                continue

            feats = gc.extract_features(info)
            features_list.append(feats)
            targets["L"].append(L)
            targets["R"].append(R)
            targets["T"].append(T)
            targets["B"].append(B)
            eval_records.append({
                "features": feats,
                "person_box": (x1, y1, x2, y2),
                "W": W, "H": H,
                "gt_box": (gt_left, gt_top, gt_right, gt_bottom),
            })
            stats["used"] += 1

            if i % 50 == 0:
                print(f"[{i}/{len(rows)}] 処理中... (使用可能: {stats['used']})")
    finally:
        landmarker.close()

    print("\n--- データ集計 ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    n = stats["used"]
    if n < 30:
        raise SystemExit(
            f"学習に使えるデータが{n}件しかありません(最低30件程度を推奨)。"
            "手動クロップ済みの写真数、またはimages-dir/pathの指定を確認してください。"
        )

    X = np.array([[f[name] for name in gc.FEATURE_NAMES] for f in features_list])
    idx_all = np.arange(n)
    idx_train, idx_test = train_test_split(idx_all, test_size=args.test_size, random_state=42)

    model_bundle = {
        "feature_names": gc.FEATURE_NAMES,
        "n_samples": n,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    print("\n--- 学習 ---")
    for key in ("L", "R", "T", "B"):
        y = np.array(targets[key])
        reg = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        )
        reg.fit(X[idx_train], y[idx_train])
        pred_test = reg.predict(X[idx_test])
        mae = float(np.mean(np.abs(pred_test - y[idx_test])))
        print(f"  {key}: MAE(holdout) = {mae:.4f}  (person幅/高さに対する比率)")
        model_bundle[key] = reg

    print("\n--- 実クロップ枠とのIoU(ホールドアウト) ---")
    ious = []
    for j in idx_test:
        rec = eval_records[j]
        margins = gc.predict_margins_model(rec["features"], model_bundle)
        x1, y1, x2, y2 = rec["person_box"]
        W, H = rec["W"], rec["H"]
        pred_crop = gc.margins_to_crop_box(
            {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "W": W, "H": H}, margins
        )
        pred_box = (
            pred_crop["CropLeft"] * W, pred_crop["CropTop"] * H,
            pred_crop["CropRight"] * W, pred_crop["CropBottom"] * H,
        )
        ious.append(iou(pred_box, rec["gt_box"]))
    print(f"  平均IoU: {np.mean(ious):.3f}  (件数: {len(ious)})")

    with open(args.output_model, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"\nモデルを保存しました: {args.output_model}")
    print("crop_calculator.py --mode general --model "
          f"{args.output_model} で使用できます。")


if __name__ == "__main__":
    main()
