# -*- coding: utf-8 -*-
"""
crop_calculator.py

人物写真(JPG)のフォルダを解析し、Lightroom Classic用のクロップ情報CSVを生成する。

- 元のJPGファイルには一切書き込みを行わない(読み込みのみ)
- 出力アスペクト比は元画像と同一(ズームイン/ズームアウトのみでクロップ)
- 人物を水平方向中央に配置
- 全身写真でない場合: 上部の余分な余白を削除しつつ、上部余白を約8%確保
- 全身写真の場合: 人物が切れないよう上下に約8%の余白を確保

【EXIF Orientation対応】
Lightroom(Adobe Camera Raw)の crs:CropTop/Left/Right/Bottom は、
EXIFのOrientationタグによる回転を適用する"前"の、保存時のピクセル座標系を
基準に解釈される。一方、人物検出は「表示向き(見たまま)」の画像に対して
行う必要があるため、本スクリプトは以下の手順で処理する:

  1. PILでEXIF Orientationを読み取り、保存時の寸法(raw_W, raw_H)を記録
  2. ImageOps.exif_transpose() で表示向きに回転補正した画像を作成し、
     その上で人物検出・クロップ計算(水平中央配置など)を行う
  3. 計算されたクロップ範囲(表示向き座標)を、Orientationタグに応じて
     保存時の座標系に変換してからCSVに出力する

MediaPipe Tasks API (PoseLandmarker) を使用します。
初回実行時にポーズ検出モデル(.task ファイル、約30MB)を自動ダウンロードします。
インターネット接続が必要です。

使い方:
    python crop_calculator.py --input /path/to/jpgs --output crop_data.csv

出力CSV列:
    filename, CropTop, CropLeft, CropRight, CropBottom, full_body, status

CropTop/Left/Right/Bottom は Lightroom の crs:CropTop 等と同じ定義で、
画像の保存時ピクセル配置を 0.0-1.0 とした場合の「クロップ枠の各辺の位置」。
"""

import argparse
import csv
import os
import urllib.request

import numpy as np
from PIL import Image, ImageOps
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ----------------------------------------------------------------------
# 調整可能パラメータ
# ----------------------------------------------------------------------
MARGIN_RATIO = 0.07          # 目標とする余白の割合(上下左右共通のベース値)
VISIBILITY_THRESHOLD = 0.5   # ランドマークを「見えている」とみなす信頼度の閾値
HEAD_PAD_FACTOR = 0.55       # 目・耳のランドマークから頭頂部を推定する際の係数(肩幅に対する比率)
FOOT_PAD_RATIO = 0.02        # 足元(全身時)にわずかに足す余白(人物身長に対する比率)

# --- 全身判定用のパラメータ ---
# MediaPipe Poseは、実際には画面外にある足首・かかとの位置も「推測」して出力することがあり、
# その推測値のvisibilityスコアが中程度でも閾値を超えてしまうことがある。
# そのため通常のVISIBILITY_THRESHOLDより厳しい基準を使い、かつ
# 「正規化座標が実際に画像の枠内(0.0〜1.0)に収まっているか」もあわせてチェックする。
FULL_BODY_VIS_THRESHOLD = 0.7       # 足首・膝の visibility がこれ以上
FULL_BODY_PRESENCE_THRESHOLD = 0.7  # 足首・膝の presence(画像内に実在する確信度)がこれ以上
FULL_BODY_FRAME_TOLERANCE = 0.0     # 正規化座標がこの許容誤差を超えて枠外ならNG(0.0=枠内必須)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_full.task")

# BlazePose 33ランドマークのインデックス(mediapipe公式の並び順)
NOSE = 0
LEFT_EYE, RIGHT_EYE = 2, 5
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

HEAD_LANDMARKS = [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]
SHOULDER_LANDMARKS = [LEFT_SHOULDER, RIGHT_SHOULDER]
HIP_LANDMARKS = [LEFT_HIP, RIGHT_HIP]
FOOT_LANDMARKS = [LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX]
ALL_LANDMARKS = list(range(33))


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"ポーズ検出モデルをダウンロード中... -> {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("ダウンロード完了。")


def create_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def get_exif_orientation(pil_img):
    try:
        exif = pil_img.getexif()
        o = exif.get(274, 1) if exif else 1
    except Exception:
        o = 1
    if o not in range(1, 9):
        o = 1
    return o


def analyze_image(path, landmarker):
    """画像を解析し、人物のバウンディングボックス(px、表示向き座標)と全身判定を返す。

    戻り値には EXIF Orientation と保存時(raw)の寸法も含める。
    """
    try:
        pil_img = Image.open(path)
    except Exception:
        return None

    orientation = get_exif_orientation(pil_img)
    raw_W, raw_H = pil_img.size  # 保存時(回転補正前)のピクセル寸法

    disp_img = ImageOps.exif_transpose(pil_img).convert("RGB")  # 表示向きに回転補正
    W, H = disp_img.size  # 表示向きの寸法(人物検出・クロップ計算に使用)

    img_rgb = np.array(disp_img)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return None

    lm = result.pose_landmarks[0]  # 最も信頼度の高い1人分

    def px(idx):
        p = lm[idx]
        return p.x * W, p.y * H, p.visibility

    def raw(idx):
        p = lm[idx]
        # presence属性が無い環境向けのフォールバック
        presence = getattr(p, "presence", 1.0)
        return p.x, p.y, p.visibility, presence

    def confidently_in_frame(idx, vis_th, pres_th, tol):
        x, y, v, pr = raw(idx)
        return (
            v >= vis_th
            and pr >= pres_th
            and -tol <= x <= 1.0 + tol
            and -tol <= y <= 1.0 + tol
        )

    visible_pts = []
    for idx in ALL_LANDMARKS:
        x, y, v = px(idx)
        if v >= VISIBILITY_THRESHOLD:
            visible_pts.append((x, y))

    if not visible_pts:
        return None

    xs = [p[0] for p in visible_pts]
    ys = [p[1] for p in visible_pts]

    x1 = min(xs)
    x2 = max(xs)

    head_ys = [px(idx)[1] for idx in HEAD_LANDMARKS if px(idx)[2] >= VISIBILITY_THRESHOLD]
    shoulder_pts = [px(idx) for idx in SHOULDER_LANDMARKS if px(idx)[2] >= VISIBILITY_THRESHOLD]

    if head_ys:
        top_head_y = min(head_ys)
        if len(shoulder_pts) == 2:
            shoulder_width = abs(shoulder_pts[0][0] - shoulder_pts[1][0])
        else:
            shoulder_width = (x2 - x1) * 0.6
        y1 = top_head_y - shoulder_width * HEAD_PAD_FACTOR
    else:
        y1 = min(ys)

    # 全身判定: 「(左右どちらかの)足首」と「(左右どちらかの)膝」の両方が、
    # 高いvisibility/presenceかつ画像枠内に収まっている場合のみ全身とみなす。
    # 足首だけを見ると、実際は画面外にある足を推測して高スコアを出すことがあるため、
    # 膝も条件に含めることで誤検出を減らす。
    def landmark_ok(idx):
        return confidently_in_frame(
            idx, FULL_BODY_VIS_THRESHOLD, FULL_BODY_PRESENCE_THRESHOLD, FULL_BODY_FRAME_TOLERANCE
        )

    ankle_ok = landmark_ok(LEFT_ANKLE) or landmark_ok(RIGHT_ANKLE)
    knee_ok = landmark_ok(LEFT_KNEE) or landmark_ok(RIGHT_KNEE)
    full_body = ankle_ok and knee_ok

    if full_body:
        foot_candidates = [LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX]
        foot_ys = [
            px(idx)[1] for idx in foot_candidates
            if confidently_in_frame(idx, FULL_BODY_VIS_THRESHOLD, FULL_BODY_PRESENCE_THRESHOLD, FULL_BODY_FRAME_TOLERANCE)
        ]
        if not foot_ys:
            # 念のためのフォールバック(理論上ankle_okがTrueなら空にならないはず)
            foot_ys = [px(idx)[1] for idx in [LEFT_ANKLE, RIGHT_ANKLE] if landmark_ok(idx)]
        bottom_y = max(foot_ys)
        person_h_est = bottom_y - y1
        bottom_y += person_h_est * FOOT_PAD_RATIO
        y2 = bottom_y
    else:
        # 全身でない場合: 「もっとアップ」な半身ポートレート風のクロップにするため、
        # 腰(ヒップ)のランドマークが見えていればそこを下端の目安にする。
        # (脚やスカートが写っていても、そこまでは含めない)
        # ヒップが見えていない場合(バストアップのみの写真等)は、
        # 実際に見えている範囲(max(ys))をそのまま使う。
        hip_ys = [px(idx)[1] for idx in HIP_LANDMARKS if px(idx)[2] >= VISIBILITY_THRESHOLD]
        visible_bottom = max(ys)
        if hip_ys:
            hip_y = max(hip_ys)  # 両ヒップのうち低い(y値が大きい)方
            y2 = min(hip_y, visible_bottom)
        else:
            y2 = visible_bottom

        # 横幅(x1,x2)も、採用した縦範囲(y1〜y2)内にあるランドマークだけで
        # 再計算する(脚や手など、クロップ範囲外の点に幅を引っ張られないようにする)
        pts_in_range = [(x, y) for (x, y) in visible_pts if y <= y2 + 1e-6]
        if pts_in_range:
            xs_in_range = [p[0] for p in pts_in_range]
            x1 = min(xs_in_range)
            x2 = max(xs_in_range)

    # 水平方向の中央位置の基準: 鼻を最優先し、見えない場合は目→肩→バウンディングボックス中心の順にフォールバック
    nose_x, nose_v = px(NOSE)[0], px(NOSE)[2]
    if nose_v >= VISIBILITY_THRESHOLD:
        center_x = nose_x
    else:
        eye_pts = [px(idx) for idx in (LEFT_EYE, RIGHT_EYE) if px(idx)[2] >= VISIBILITY_THRESHOLD]
        if eye_pts:
            center_x = sum(p[0] for p in eye_pts) / len(eye_pts)
        elif len(shoulder_pts) == 2:
            center_x = (shoulder_pts[0][0] + shoulder_pts[1][0]) / 2.0
        else:
            center_x = (x1 + x2) / 2.0

    y1 = max(0.0, y1)
    y2 = min(float(H), y2)
    x1 = max(0.0, x1)
    x2 = min(float(W), x2)

    if x2 <= x1 or y2 <= y1:
        return None

    return {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2, "W": W, "H": H,
        "full_body": full_body, "center_x": center_x,
        "orientation": orientation, "raw_W": raw_W, "raw_H": raw_H,
    }


def compute_crop(info):
    """人物情報からクロップ矩形を計算し、Lightroom式の0-1割合を返す。"""
    W, H = info["W"], info["H"]
    x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]
    person_w = x2 - x1
    person_h = y2 - y1
    cx = info.get("center_x", (x1 + x2) / 2.0)

    aspect = W / H

    wc_from_width = person_w / (1 - 2 * MARGIN_RATIO)

    if info["full_body"]:
        # 上下とも約8%の余白を確保する必要がある高さ
        hc_from_height = person_h / (1 - 2 * MARGIN_RATIO)
    else:
        # 上部のみ約8%の余白を確保しつつ、人物の見えている範囲(y1〜y2)を
        # 必ず収める高さ(下側は余白なしでちょうど収まる想定)
        hc_from_height = person_h / (1 - MARGIN_RATIO)

    wc_from_height = hc_from_height * aspect
    # 人物が絶対に切れないよう、横幅基準・高さ基準のうち大きい方を採用
    Wc = max(wc_from_width, wc_from_height)

    # 中央配置の基準(鼻など)が人物のバウンディングボックス中心とズレている場合、
    # そのズレを考慮しても左右とも人物が切れず、かつ約8%以上の余白を確保できる幅を保証する
    half_needed = max(x2 - cx, cx - x1) / (1 - MARGIN_RATIO)
    Wc = max(Wc, 2 * half_needed)

    Wc = min(Wc, W)
    Hc = Wc / aspect
    if Hc > H:
        Hc = H
        Wc = Hc * aspect

    left = cx - Wc / 2.0
    left = max(0.0, min(left, W - Wc))
    right = left + Wc

    if info["full_body"]:
        top = y1 - (Hc - person_h) / 2.0
    else:
        top = y1 - MARGIN_RATIO * Hc

    top = max(0.0, min(top, H - Hc))
    bottom = top + Hc

    return {
        "CropLeft": left / W,
        "CropRight": right / W,
        "CropTop": top / H,
        "CropBottom": bottom / H,
    }


def display_crop_to_raw(crop, orientation):
    """表示向き(EXIF回転補正後)のクロップ割合を、Lightroomが解釈する
    保存時(raw, 回転補正前)の座標系のクロップ割合に変換する。

    Lightroomの crs:CropTop/Left/Right/Bottom は、EXIF Orientationによる
    回転を適用する前のピクセル配置を基準にしているため、この変換が必須。
    """
    dLeft, dTop = crop["CropLeft"], crop["CropTop"]
    dRight, dBottom = crop["CropRight"], crop["CropBottom"]

    def inv(u, v):
        # 表示向き正規化座標 (u,v) -> 保存時(raw)正規化座標
        if orientation == 1:
            return u, v
        elif orientation == 2:      # 左右反転
            return 1 - u, v
        elif orientation == 3:      # 180度回転
            return 1 - u, 1 - v
        elif orientation == 4:      # 上下反転
            return u, 1 - v
        elif orientation == 5:      # 転置(主対角線)
            return v, u
        elif orientation == 6:      # 時計回りに90度回転して表示
            return v, 1 - u
        elif orientation == 7:      # 転置(反対角線)
            return 1 - v, 1 - u
        elif orientation == 8:      # 反時計回りに90度回転して表示
            return 1 - v, u
        else:
            return u, v

    u1, v1 = inv(dLeft, dTop)
    u2, v2 = inv(dRight, dBottom)

    return {
        "CropLeft": min(u1, u2),
        "CropRight": max(u1, u2),
        "CropTop": min(v1, v2),
        "CropBottom": max(v1, v2),
    }


def main():
    parser = argparse.ArgumentParser(description="人物写真のクロップ情報CSVを生成")
    parser.add_argument("--input", required=True, help="JPG画像が入ったフォルダ")
    parser.add_argument("--output", default="crop_data.csv", help="出力CSVファイルパス")
    parser.add_argument("--recursive", action="store_true", help="サブフォルダも再帰的に処理")
    args = parser.parse_args()

    ensure_model()

    if args.recursive:
        image_paths = []
        for root, _, files in os.walk(args.input):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg")):
                    image_paths.append(os.path.join(root, f))
    else:
        image_paths = [
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith((".jpg", ".jpeg"))
        ]

    image_paths.sort()
    print(f"対象画像: {len(image_paths)} 件")

    rows = []
    landmarker = create_landmarker()
    try:
        for i, path in enumerate(image_paths, 1):
            filename = os.path.basename(path)
            info = analyze_image(path, landmarker)
            if info is None:
                rows.append({
                    "filename": filename,
                    "CropTop": "", "CropLeft": "", "CropRight": "", "CropBottom": "",
                    "full_body": "", "status": "NO_PERSON_DETECTED",
                })
                print(f"[{i}/{len(image_paths)}] {filename}: 人物検出失敗")
                continue

            crop_display = compute_crop(info)
            crop_raw = display_crop_to_raw(crop_display, info["orientation"])
            rows.append({
                "filename": filename,
                "CropTop": f"{crop_raw['CropTop']:.6f}",
                "CropLeft": f"{crop_raw['CropLeft']:.6f}",
                "CropRight": f"{crop_raw['CropRight']:.6f}",
                "CropBottom": f"{crop_raw['CropBottom']:.6f}",
                "full_body": "1" if info["full_body"] else "0",
                "status": "OK",
            })
            print(f"[{i}/{len(image_paths)}] {filename}: OK (full_body={info['full_body']}, orientation={info['orientation']})")
    finally:
        landmarker.close()

    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "CropTop", "CropLeft", "CropRight", "CropBottom",
            "full_body", "status",
        ])
        writer.writeheader()
        writer.writerows(rows)

    ok_count = sum(1 for r in rows if r["status"] == "OK")
    print(f"\n完了: {ok_count}/{len(rows)} 件成功。出力先: {args.output}")


if __name__ == "__main__":
    main()
