# -*- coding: utf-8 -*-
"""
evaluate_crop.py

手動クロップ実績(ExportCropHistory.lua の出力CSV)を正解として、
自動クロップの精度を測定・検証するツール。

主な用途:

1. 現状の精度を測る
       python evaluate_crop.py --crop-csv hand_crop.csv

2. パラメータを掃引して最適値を探す
       python evaluate_crop.py --crop-csv hand_crop.csv --sweep MARGIN_RATIO=0.08,0.09,0.10,0.11

3. 交差検証で「調整に使っていないデータ」での性能を測る(過学習の検出)
       python evaluate_crop.py --crop-csv hand_crop.csv --cross-validate \
           --sweep MARGIN_RATIO=0.08,0.09,0.10,0.11

【なぜ交差検証が必要か】
定数を手元のデータに合わせて調整し、同じデータで評価すると、実際には汎化して
いなくても数値が良く見える(過学習)。本ツールは写真を連番ブロック単位で分割し、
一部のブロックで最適値を決めて、残りのブロックで評価する。
連続する写真は同じ衣装・セットアップでほぼ同一構図になるため、ランダム分割では
学習側と評価側にほぼ同じ写真が入ってしまい、検証にならない。

【キャッシュ】
人物検出が処理時間のほぼ全てを占めるため、検出結果(全員分のランドマーク)を
pickleにキャッシュする。パラメータを変えて何度も評価する場合、2回目以降は
検出をやり直さないため一瞬で終わる。crop_calculator.py の person_box() /
combine_person_boxes() をそのまま呼ぶので、本体とロジックが乖離することはない。
--refresh-cache でキャッシュを作り直せる。
"""

import argparse
import csv
import os
import pickle
import re
import statistics

from PIL import Image, ImageOps

import crop_calculator as cc

CACHE_VERSION = 1
# 交差検証で使うブロックの大きさ(連番何枚を1ブロックとみなすか)
DEFAULT_BLOCK_SIZE = 25


class _Pt:
    """キャッシュしたランドマークを、mediapipe のランドマークと同じ形で扱うための入れ物。"""
    __slots__ = ("x", "y", "visibility", "presence")

    def __init__(self, x, y, visibility, presence):
        self.x, self.y, self.visibility, self.presence = x, y, visibility, presence


def load_hand_crops(path):
    """手動クロップCSVを読み込む。回転クロップは本ツールの対象外なので除外する。"""
    rows = []
    skipped_rotated = 0
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("is_cropped") != "1":
                continue
            if abs(float(r.get("CropAngle") or 0)) > 0.01:
                skipped_rotated += 1
                continue
            rows.append(r)
    return rows, skipped_rotated


def build_cache(rows, cache_path, refresh=False):
    """人物検出の結果をキャッシュする(既にあれば再利用)。"""
    if os.path.exists(cache_path) and not refresh:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if data.get("version") == CACHE_VERSION and data.get("n_rows") == len(rows):
            print(f"検出キャッシュを再利用: {cache_path} ({len(data['items'])}件)")
            return data["items"]
        print("キャッシュが古いか件数が変わっているため作り直します。")

    cc.ensure_model()
    detector = cc.create_landmarker()
    items = []
    try:
        for i, r in enumerate(rows, 1):
            try:
                pil = Image.open(r["path"])
            except Exception:
                continue
            orientation = cc.get_exif_orientation(pil)
            disp = ImageOps.exif_transpose(pil).convert("RGB")
            W, H = disp.size
            poses, stage = detector.detect_pose(disp)
            if not poses:
                continue
            hand = cc.raw_crop_to_display(
                {k: float(r[k]) for k in ("CropTop", "CropLeft", "CropRight", "CropBottom")},
                orientation,
            )
            items.append({
                "filename": r["filename"],
                "seq": _seq_of(r["filename"]),
                "W": W, "H": H,
                "poses": [[(p.x, p.y, p.visibility, getattr(p, "presence", 1.0)) for p in ps]
                          for ps in poses],
                "detect_stage": stage,
                "hand": hand,
            })
            if i % 100 == 0:
                print(f"  検出中... {i}/{len(rows)}")
    finally:
        detector.close()

    with open(cache_path, "wb") as f:
        pickle.dump({"version": CACHE_VERSION, "n_rows": len(rows), "items": items}, f)
    print(f"検出キャッシュを作成: {cache_path} ({len(items)}件)")
    return items


def _seq_of(filename):
    """ファイル名から連番を取り出す(ブロック分割に使う)。"""
    m = re.findall(r"(\d+)", filename)
    return int(m[-1]) if m else 0


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate(items):
    """各写真について、自動クロップと手動クロップのIoUを求める。

    person_box() / combine_person_boxes() / compute_crop() は crop_calculator の
    ものをそのまま呼ぶため、本体の挙動がそのまま評価される。
    """
    results = []
    for it in items:
        W, H = it["W"], it["H"]
        poses = [[_Pt(*p) for p in ps] for ps in it["poses"]]
        info = cc.combine_person_boxes([cc.person_box(p, W, H) for p in poses], W, H)
        if info is None:
            continue
        crop = cc.compute_crop(info)
        hd = it["hand"]
        v = iou(
            (hd["CropLeft"] * W, hd["CropTop"] * H, hd["CropRight"] * W, hd["CropBottom"] * H),
            (crop["CropLeft"] * W, crop["CropTop"] * H, crop["CropRight"] * W, crop["CropBottom"] * H),
        )
        # 人物がクロップから外れていないか(1px の余裕をみる)
        cut = (info["x1"] < crop["CropLeft"] * W - 1 or info["x2"] > crop["CropRight"] * W + 1
               or info["y1"] < crop["CropTop"] * H - 1 or info["y2"] > crop["CropBottom"] * H + 1)
        results.append({
            "filename": it["filename"], "seq": it["seq"], "iou": v,
            "full_body": info["full_body"], "n_persons": info["n_persons"],
            "detect_stage": it["detect_stage"], "cut": cut,
        })
    return results


def summarize(results, label="全体"):
    if not results:
        print(f"{label}: 対象なし")
        return None
    v = sorted(r["iou"] for r in results)
    mean = statistics.mean(v)
    print(f"{label}: {len(v)}枚  平均 {mean:.4f}  中央値 {statistics.median(v):.4f}  "
          f"下位10% {v[max(0, len(v)//10 - 1)]:.4f}  最小 {min(v):.4f}  "
          f"人物が切れた枚数 {sum(1 for r in results if r['cut'])}")
    return mean


def print_breakdown(results):
    summarize(results, "全体          ")
    summarize([r for r in results if r["full_body"]], "  全身        ")
    summarize([r for r in results if not r["full_body"]], "  半身        ")
    summarize([r for r in results if r["n_persons"] > 1], "  複数人      ")
    summarize([r for r in results if r["detect_stage"] not in (0, None)], "  検出再試行  ")


def parse_sweep(spec):
    """'MARGIN_RATIO=0.08,0.09' 形式を (定数名, [値...]) に変換する。"""
    name, values = spec.split("=", 1)
    name = name.strip()
    if not hasattr(cc, name):
        raise SystemExit(f"crop_calculator に定数 {name} がありません")
    return name, [float(x) for x in values.split(",")]


def sweep(items, name, values):
    original = getattr(cc, name)
    best = None
    print(f"\n{name} の掃引")
    try:
        for val in values:
            setattr(cc, name, val)
            res = evaluate(items)
            mean = statistics.mean([r["iou"] for r in res])
            mark = "  <- 現在の設定" if abs(val - original) < 1e-12 else ""
            print(f"  {val:<8} 平均IoU {mean:.4f}{mark}")
            if best is None or mean > best[1]:
                best = (val, mean)
    finally:
        setattr(cc, name, original)
    print(f"  最良: {name}={best[0]} (平均IoU {best[1]:.4f})")
    return best


def blocks_of(items, block_size):
    """連番をブロックにまとめる。連続する写真は構図が似ているため、
    ブロックごと学習側/評価側に振り分けないと検証にならない。"""
    groups = {}
    for i, it in enumerate(items):
        groups.setdefault(it["seq"] // block_size, []).append(i)
    return [sorted(v) for _, v in sorted(groups.items())]


def cross_validate(items, name, values, block_size, folds):
    """ブロック単位のk分割交差検証。

    各分割で「学習側だけで最適値を決め」「評価側で測る」ことで、
    調整に使っていないデータに対する性能を見る。
    """
    bl = blocks_of(items, block_size)
    print(f"\n交差検証: {len(bl)}ブロック を {folds}分割 (ブロックサイズ {block_size}枚)")
    original = getattr(cc, name)
    tuned_scores, fixed_scores, picked = [], [], []
    try:
        for k in range(folds):
            test_idx = [i for bi, b in enumerate(bl) if bi % folds == k for i in b]
            train_idx = [i for bi, b in enumerate(bl) if bi % folds != k for i in b]
            if not test_idx or not train_idx:
                continue
            train = [items[i] for i in train_idx]
            test = [items[i] for i in test_idx]

            # 学習側で最適値を決める
            best_val, _ = None, None
            best_mean = -1
            for val in values:
                setattr(cc, name, val)
                m = statistics.mean([r["iou"] for r in evaluate(train)])
                if m > best_mean:
                    best_mean, best_val = m, val
            # 評価側で測る
            setattr(cc, name, best_val)
            tuned = statistics.mean([r["iou"] for r in evaluate(test)])
            setattr(cc, name, original)
            fixed = statistics.mean([r["iou"] for r in evaluate(test)])
            tuned_scores.append(tuned)
            fixed_scores.append(fixed)
            picked.append(best_val)
            print(f"  分割{k+1}: 学習側の最適値 {name}={best_val}  "
                  f"評価側 {tuned:.4f} (現在の設定なら {fixed:.4f})")
    finally:
        setattr(cc, name, original)

    print(f"\n  評価側の平均: 学習側で選んだ値 {statistics.mean(tuned_scores):.4f} / "
          f"現在の設定({original}) {statistics.mean(fixed_scores):.4f}")
    if len(set(picked)) == 1 and picked[0] == original:
        print(f"  → どの分割でも {name}={original} が選ばれた。現在の設定は過学習ではない。")
    else:
        print(f"  → 分割ごとに選ばれた値: {picked}")
        print("     ばらつきが大きい場合、その定数はデータに敏感で過学習しやすい。")


def main():
    ap = argparse.ArgumentParser(description="手動クロップ実績に対する自動クロップの精度を検証")
    ap.add_argument("--crop-csv", required=True, help="ExportCropHistory.lua が出力したCSV")
    ap.add_argument("--cache", default="eval_cache.pkl", help="人物検出結果のキャッシュ")
    ap.add_argument("--refresh-cache", action="store_true", help="キャッシュを作り直す")
    ap.add_argument("--sweep", help="掃引する定数。例: MARGIN_RATIO=0.08,0.09,0.10")
    ap.add_argument("--cross-validate", action="store_true", help="ブロック単位の交差検証を行う")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    ap.add_argument("--worst", type=int, default=0, help="一致度が低い写真を上位N件表示する")
    args = ap.parse_args()

    rows, rotated = load_hand_crops(args.crop_csv)
    print(f"手動クロップ: {len(rows)}枚 (回転クロップ {rotated}枚は対象外)")
    items = build_cache(rows, args.cache, args.refresh_cache)

    print()
    print("=== 現在の設定での精度 ===")
    print(f"MARGIN_RATIO={cc.MARGIN_RATIO}  FOOT_PAD_RATIO={cc.FOOT_PAD_RATIO}  "
          f"FULL_BODY_VIS={cc.FULL_BODY_VIS_THRESHOLD}  MAX_PERSONS={cc.MAX_PERSONS}")
    results = evaluate(items)
    print_breakdown(results)

    if args.worst:
        print(f"\n一致度が低い写真 上位{args.worst}件:")
        for r in sorted(results, key=lambda r: r["iou"])[:args.worst]:
            print(f"  {r['filename']}  IoU {r['iou']:.3f}  "
                  f"全身={r['full_body']} 人数={r['n_persons']} 検出段階={r['detect_stage']}")

    if args.sweep:
        name, values = parse_sweep(args.sweep)
        sweep(items, name, values)
        if args.cross_validate:
            cross_validate(items, name, values, args.block_size, args.folds)
    elif args.cross_validate:
        raise SystemExit("--cross-validate は --sweep と一緒に指定してください")


if __name__ == "__main__":
    main()
