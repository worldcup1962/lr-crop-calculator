# -*- coding: utf-8 -*-
"""
debug_single.py

1枚の画像に対してポーズ検出を実行し、
- 各ランドマークの座標・visibility・presence
- 計算されたバウンディングボックス(表示向き座標)
- full_body判定
- 表示向きクロップ矩形、および実際にCSVへ出力される raw(保存時)座標系のクロップ
をコンソールに出力し、あわせて検出結果を描画した確認用画像を保存します。

確認用画像は「表示向き(EXIF回転補正後、見たままの向き)」で描画されます。
実際にLightroomへ渡すCSVの値は raw 座標系である点に注意してください。

crop_calculator.py と同じく --mode promo/general に対応しています。
general モードでは、判定に使った特徴量(視線方向・背景密度など)と
予測された余白(L/R/T/B)もあわせて表示します。

使い方:
    python debug_single.py --input path/to/photo.jpg --output debug_out.jpg
    python debug_single.py --input path/to/photo.jpg --output debug_out.jpg \
        --mode general --model general_crop_model.pkl
"""

import argparse
from PIL import Image, ImageDraw

import crop_calculator as cc


LANDMARK_NAMES = {
    0: "NOSE", 1: "LEFT_EYE_INNER", 2: "LEFT_EYE", 3: "LEFT_EYE_OUTER",
    4: "RIGHT_EYE_INNER", 5: "RIGHT_EYE", 6: "RIGHT_EYE_OUTER",
    7: "LEFT_EAR", 8: "RIGHT_EAR", 9: "MOUTH_LEFT", 10: "MOUTH_RIGHT",
    11: "LEFT_SHOULDER", 12: "RIGHT_SHOULDER", 13: "LEFT_ELBOW", 14: "RIGHT_ELBOW",
    15: "LEFT_WRIST", 16: "RIGHT_WRIST", 17: "LEFT_PINKY", 18: "RIGHT_PINKY",
    19: "LEFT_INDEX", 20: "RIGHT_INDEX", 21: "LEFT_THUMB", 22: "RIGHT_THUMB",
    23: "LEFT_HIP", 24: "RIGHT_HIP", 25: "LEFT_KNEE", 26: "RIGHT_KNEE",
    27: "LEFT_ANKLE", 28: "RIGHT_ANKLE", 29: "LEFT_HEEL", 30: "RIGHT_HEEL",
    31: "LEFT_FOOT_INDEX", 32: "RIGHT_FOOT_INDEX",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="debug_out.jpg")
    parser.add_argument("--mode", choices=["promo", "general"], default="promo")
    parser.add_argument("--model", default=None,
                         help="general モード用の学習済みモデル(省略時はスクリプトと同じ"
                              "フォルダの general_crop_model.pkl を探す。無ければヒューリスティック)")
    args = parser.parse_args()

    general_model = None
    if args.mode == "general":
        import os
        import general_crop
        model_path = args.model or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "general_crop_model.pkl"
        )
        general_model = general_crop.load_model(model_path)
        if general_model is None:
            print(f"[general] 学習済みモデルが見つかりません({model_path})。ヒューリスティックを使用します。")
        else:
            print(f"[general] 学習済みモデルを読み込みました: {model_path}")

    cc.ensure_model()

    pil_img = Image.open(args.input)
    orientation = cc.get_exif_orientation(pil_img)
    raw_W, raw_H = pil_img.size
    print(f"保存時(raw)サイズ: W={raw_W}, H={raw_H}   EXIF Orientation={orientation}")

    landmarker = cc.create_landmarker()
    try:
        info = cc.analyze_image(args.input, landmarker)
    finally:
        landmarker.close()

    if info is None:
        print("!! 人物が検出されませんでした")
        return

    W, H = info["W"], info["H"]
    print(f"表示向き(回転補正後)サイズ: W={W}, H={H}")

    stage = info["detect_stage"]
    if stage == 0:
        print("検出: 通常条件(段階0)で成功")
    else:
        max_dim, conf, contrast = cc.DETECT_STAGES[stage]
        print(f"検出: 通常条件では失敗し、フォールバック段階{stage}で成功 "
              f"(最大辺={max_dim}, 閾値={conf}, コントラスト強調={contrast})")
        print("  ※ この写真は検出精度が落ちている可能性があるため、結果の確認を推奨します")

    # analyze_image が実際に使ったランドマーク・表示向き画像をそのまま使う
    disp_img = info["disp_img"]
    lm = info["landmarks"]
    print("\n--- 全ランドマーク (idx: 名前  x_norm, y_norm  visibility  presence) ---")
    for idx in range(33):
        p = lm[idx]
        presence = getattr(p, "presence", None)
        name = LANDMARK_NAMES.get(idx, str(idx))
        print(f"{idx:2d} {name:16s} x={p.x:.4f} y={p.y:.4f}  vis={p.visibility:.3f}  pres={presence}")

    print("\n--- analyze_image() の結果 ---")
    print(f"x1={info['x1']:.1f} y1={info['y1']:.1f} x2={info['x2']:.1f} y2={info['y2']:.1f}")
    print(f"person_w={info['x2']-info['x1']:.1f}  person_h={info['y2']-info['y1']:.1f}")
    print(f"full_body = {info['full_body']}")

    if args.mode == "general":
        crop_display, features, margins = general_crop.compute_crop_general(info, general_model)
        print("\n--- generalモード: 特徴量 ---")
        for k, v in features.items():
            print(f"  {k}: {v:.4f}")
        print("--- generalモード: 予測された余白(person幅/高さに対する比率) ---")
        print(f"  L={margins['L']:.4f}  R={margins['R']:.4f}  T={margins['T']:.4f}  B={margins['B']:.4f}")
    else:
        crop_display = cc.compute_crop(info)
    crop_raw = cc.display_crop_to_raw(crop_display, orientation)

    print(f"\n--- 表示向き座標でのクロップ(0-1割合、mode={args.mode}) ---")
    print(crop_display)
    print("--- CSVに出力される raw(保存時)座標でのクロップ(0-1割合) ---")
    print(crop_raw)

    Wc = (crop_display["CropRight"] - crop_display["CropLeft"]) * W
    Hc = (crop_display["CropBottom"] - crop_display["CropTop"]) * H
    print(f"\nクロップ実寸(表示向きpx): Wc={Wc:.1f}, Hc={Hc:.1f}"
          f"  (元画像に対する割合: {Wc/W*100:.1f}% x {Hc/H*100:.1f}%)")

    # 描画は「表示向き(見たまま)」の画像に対して行う
    vis = disp_img.copy()
    draw = ImageDraw.Draw(vis)

    # バウンディングボックス(緑) = 検出範囲
    draw.rectangle([info["x1"], info["y1"], info["x2"], info["y2"]], outline=(0, 255, 0), width=4)

    # クロップ矩形(赤) = 表示向き座標でのクロップ範囲
    cl = crop_display["CropLeft"] * W
    cr = crop_display["CropRight"] * W
    ct = crop_display["CropTop"] * H
    cb = crop_display["CropBottom"] * H
    draw.rectangle([cl, ct, cr, cb], outline=(255, 0, 0), width=6)

    # ランドマーク点(黄色、visibility>=0.3)
    for idx in range(33):
        p = lm[idx]
        if p.visibility >= 0.3:
            x, y = p.x * W, p.y * H
            r = 10
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 0))

    vis.save(args.output, quality=95)
    print(f"\n確認用画像を保存しました(表示向き、mode={args.mode}): {args.output}")
    print("  緑枠 = 検出された人物のバウンディングボックス")
    print("  赤枠 = 計算されたクロップ範囲(表示向き座標)")
    print("  黄色い点 = visibility>=0.3のランドマーク")
    print("\n※ この画像は見たまま(表示向き)の座標で描画しています。")
    print("   実際にLightroomへ渡すCSVの値は raw(保存時)座標系に変換済みの値です。")


if __name__ == "__main__":
    main()
