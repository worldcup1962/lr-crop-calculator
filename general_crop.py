# -*- coding: utf-8 -*-
"""
general_crop.py

汎用ポートレート写真クロップ機能(--mode general)のロジック。

crop_calculator.py の宣材写真用ロジック(compute_crop, promoモード)は一切変更せず、
そのまま維持する。本モジュールはそれとは独立した「別の構図方針」を提供する:

- 人物(バウンディングボックス)の中心を画像の水平中央に配置する
  (promoモードが「鼻」を基準に中央化するのに対し、こちらは体全体の中心を基準にする。
   HORIZONTAL_CENTER_WEIGHT を下げると、視線・体の向きに応じた左右非対称の配置
   ——いわゆる「空間を残す」構図——も許容できる)
- 背景の情報量(エッジ密度で近似)を見て、うるさい側の余白を詰め、
  すっきりした側の余白を広げる(クロップの「幅」に反映される)
- 手動トリミングデータ(Lightroomのクロップ履歴)があれば、上記の余白を
  ルールベースの経験則ではなく学習済み回帰モデルから予測する

依存関係: 特徴量抽出・ヒューリスティックは Pillow + numpy のみで完結する
(crop_calculator.py の追加依存を増やさない)。学習済みモデルを使う場合のみ、
pickle 経由で scikit-learn のオブジェクトを読み込むため scikit-learn が必要になる。

クロップ矩形のパラメータ化:
    人物のバウンディングボックス(x1,y1,x2,y2, 幅person_w, 高さperson_h)を基準に、
    各辺の余白をバウンディングボックスのサイズに対する比率として表す。

        L = (x1 - CropLeft) / person_w    (右に行くほど正、人物の左に余白)
        R = (CropRight - x2) / person_w
        T = (y1 - CropTop)  / person_h
        B = (CropBottom - y2) / person_h

    L,R,T,B が学習・予測される4つのターゲット。負値は「人物のバウンディングボックスの
    内側までクロップする(生え際や指先が少し切れるなど、実際の人物写真でもよくある
    タイトなクロップ)」を意味し、promoモードと違って許容する。
"""

import pickle

import numpy as np
from PIL import Image, ImageFilter

import crop_calculator as cc

# ----------------------------------------------------------------------
# 特徴量
# ----------------------------------------------------------------------
FEATURE_NAMES = [
    "aspect", "person_w_ratio", "person_h_ratio", "full_body",
    "bbox_cx_ratio", "bbox_cy_ratio", "gaze_dir",
    "density_left", "density_right", "density_top", "density_bottom", "density_overall",
]

DETAIL_MAP_MAX_DIM = 512     # エッジ密度マップ計算時の最大辺(速度のためダウンサンプル)
BAND_RATIO = 0.15            # 余白の「うるささ」を測る帯の幅(人物サイズに対する比率)

# ----------------------------------------------------------------------
# ヒューリスティック(モデル未学習時のデフォルト)のパラメータ
# ----------------------------------------------------------------------
BASE_SIDE_MARGIN = 0.12
BASE_TOP_MARGIN = 0.15
BASE_BOTTOM_MARGIN_FULL_BODY = 0.15
BASE_BOTTOM_MARGIN_HALF_BODY = 0.02
GAZE_MARGIN_BOOST = 0.10     # 視線・体の向きに応じた左右非対称の強さ
DENSITY_ADJUST_STRENGTH = 0.35
DENSITY_FACTOR_RANGE = (0.6, 1.4)
HEURISTIC_MARGIN_FLOOR = 0.02

# 水平方向の配置: 0.0 = 予測された左右の余白(L, R)のとおりに配置する(既定)
#                 1.0 = 人物(バウンディングボックス)の中心を必ず画像の水平中央に置く
# 中間値にすると、その割合で両者を混ぜた位置になる。
# 左右の余白はいずれの場合もクロップの「幅」の決定には使われる(位置だけが変わる)。
#
# 既定を 0.0 にしている理由:
# 手動クロップ実績を分析したところ、クロップ中心からのズレは
# 鼻4.2% / 肩の中点5.7% / 体bbox中心4.5%(いずれも平均)であり、
# 特定の基準を厳密に中央へ置く構図ではなく「ゆるやかに中央寄り」だった。
# 1.0 にすると体の中心をきっちり中央に固定するため、手動クロップの再現性は下がる。
HORIZONTAL_CENTER_WEIGHT = 0.0

# モデル予測値に対する安全クランプ(暴走した予測で極端なクロップにならないように)
MODEL_MARGIN_CLAMP = (-0.35, 1.5)
# どんな余白値でも、人物の幅・高さの最低50%は残す(安全弁)
MIN_BOX_FRACTION = 0.5


def _compute_detail_map(disp_img, max_dim=DETAIL_MAP_MAX_DIM):
    """画像のエッジ密度マップ(0-1に正規化)を計算する。

    本格的な顕著性(サリエンシー)モデルの代わりに、軽量なエッジ検出で
    「背景の情報量(うるささ)」を近似する。追加の重い依存(opencv-contrib等)を
    避けるため Pillow の FIND_EDGES のみを使用する。
    """
    w, h = disp_img.size
    scale = min(1.0, max_dim / float(max(w, h)))
    if scale < 1.0:
        small = disp_img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    else:
        small = disp_img

    gray = small.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    radius = max(1, int(min(small.size) * 0.01))
    edges = edges.filter(ImageFilter.GaussianBlur(radius=radius))

    arr = np.asarray(edges, dtype=np.float32)
    p95 = float(np.percentile(arr, 95)) if arr.size else 1.0
    if p95 < 1e-6:
        p95 = 1.0
    arr = np.clip(arr / p95, 0.0, 1.0)
    return arr, scale


def _band_density(arr, scale, x_min, x_max, y_min, y_max):
    h_arr, w_arr = arr.shape
    xi1 = int(np.clip(x_min * scale, 0, w_arr - 1))
    xi2 = int(np.clip(x_max * scale, xi1 + 1, w_arr))
    yi1 = int(np.clip(y_min * scale, 0, h_arr - 1))
    yi2 = int(np.clip(y_max * scale, yi1 + 1, h_arr))
    region = arr[yi1:yi2, xi1:xi2]
    if region.size == 0:
        return float(arr.mean())
    return float(region.mean())


def _gaze_dir(info):
    """人物の向き(視線・体の向き)を -1(左向き)〜+1(右向き)で近似する。

    肩の中点に対する鼻の水平方向のズレを、肩幅で正規化して求める。
    肩が両方見えない場合は 0(正面向き相当)とする。
    """
    lm = info["landmarks"]
    W, H = info["W"], info["H"]

    def px(idx):
        p = lm[idx]
        return p.x * W, p.y * H, p.visibility

    lsx, lsy, lsv = px(cc.LEFT_SHOULDER)
    rsx, rsy, rsv = px(cc.RIGHT_SHOULDER)
    if lsv < cc.VISIBILITY_THRESHOLD or rsv < cc.VISIBILITY_THRESHOLD:
        return 0.0

    shoulder_mid_x = (lsx + rsx) / 2.0
    shoulder_width = abs(lsx - rsx)
    if shoulder_width < 1e-6:
        return 0.0

    nx, ny, nv = px(cc.NOSE)
    ref_x = nx if nv >= cc.VISIBILITY_THRESHOLD else shoulder_mid_x

    return float(np.clip((ref_x - shoulder_mid_x) / (shoulder_width / 2.0), -1.0, 1.0))


def extract_features(info):
    """analyze_image() が返す info から、学習・予測共通の特徴量dictを作る。"""
    W, H = info["W"], info["H"]
    x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]
    person_w = max(1e-6, x2 - x1)
    person_h = max(1e-6, y2 - y1)

    arr, scale = _compute_detail_map(info["disp_img"])
    band_w = max(4.0, person_w * BAND_RATIO)
    band_h = max(4.0, person_h * BAND_RATIO)

    density_left = _band_density(arr, scale, x1 - band_w, x1, y1, y2)
    density_right = _band_density(arr, scale, x2, x2 + band_w, y1, y2)
    density_top = _band_density(arr, scale, x1, x2, y1 - band_h, y1)
    density_bottom = _band_density(arr, scale, x1, x2, y2, y2 + band_h)
    density_overall = float(arr.mean())

    return {
        "aspect": W / H,
        "person_w_ratio": person_w / W,
        "person_h_ratio": person_h / H,
        "full_body": 1.0 if info["full_body"] else 0.0,
        "bbox_cx_ratio": ((x1 + x2) / 2.0) / W,
        "bbox_cy_ratio": ((y1 + y2) / 2.0) / H,
        "gaze_dir": _gaze_dir(info),
        "density_left": density_left,
        "density_right": density_right,
        "density_top": density_top,
        "density_bottom": density_bottom,
        "density_overall": density_overall,
    }


# ----------------------------------------------------------------------
# 余白予測: ヒューリスティック(フォールバック)
# ----------------------------------------------------------------------
def _density_factor(side_density, overall_density):
    ratio = side_density / (overall_density + 1e-6)
    factor = 1.0 - DENSITY_ADJUST_STRENGTH * (ratio - 1.0)
    lo, hi = DENSITY_FACTOR_RANGE
    return float(np.clip(factor, lo, hi))


def heuristic_margins(features):
    """学習済みモデルがない場合のデフォルト余白(三分割法+背景密度による経験則)。"""
    full_body = features["full_body"] >= 0.5

    base_top = BASE_TOP_MARGIN
    base_bottom = BASE_BOTTOM_MARGIN_FULL_BODY if full_body else BASE_BOTTOM_MARGIN_HALF_BODY
    base_side = BASE_SIDE_MARGIN

    # 視線・体の向きに応じて左右非対称に(向いている側に「空間」を残す)
    # ※ HORIZONTAL_CENTER_WEIGHT=1.0(既定)では配置は中央固定になるため、
    #    この左右差はクロップの「幅」にのみ影響する
    delta = GAZE_MARGIN_BOOST * features["gaze_dir"]
    left = base_side - delta / 2.0
    right = base_side + delta / 2.0

    # 背景のうるささに応じて微調整(うるさい側は詰め、すっきりした側は広げる)
    overall = features["density_overall"]
    left *= _density_factor(features["density_left"], overall)
    right *= _density_factor(features["density_right"], overall)
    top = base_top * _density_factor(features["density_top"], overall)
    bottom = base_bottom * _density_factor(features["density_bottom"], overall) if full_body else base_bottom

    left = max(HEURISTIC_MARGIN_FLOOR, left)
    right = max(HEURISTIC_MARGIN_FLOOR, right)
    top = max(HEURISTIC_MARGIN_FLOOR, top)
    bottom = max(0.0, bottom)

    return {"L": left, "R": right, "T": top, "B": bottom}


# ----------------------------------------------------------------------
# 余白予測: 学習済みモデル
# ----------------------------------------------------------------------
def load_model(path):
    """学習済みモデルバンドルを読み込む。ファイルが無ければ None を返す。"""
    import os
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_margins_model(features, model_bundle):
    x = np.array([[features[name] for name in model_bundle["feature_names"]]])
    lo, hi = MODEL_MARGIN_CLAMP
    margins = {}
    for key in ("L", "R", "T", "B"):
        pred = float(model_bundle[key].predict(x)[0])
        margins[key] = float(np.clip(pred, lo, hi))
    return margins


# ----------------------------------------------------------------------
# 余白 -> クロップ矩形
# ----------------------------------------------------------------------
def margins_to_crop_box(info, margins):
    W, H = info["W"], info["H"]
    x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]
    person_w = x2 - x1
    person_h = y2 - y1
    aspect = W / H

    L, R, T, B = margins["L"], margins["R"], margins["T"], margins["B"]

    wc_from_width = max(person_w * (1.0 + L + R), person_w * MIN_BOX_FRACTION)
    hc_from_height = max(person_h * (1.0 + T + B), person_h * MIN_BOX_FRACTION)
    wc_from_height = hc_from_height * aspect

    Wc = max(wc_from_width, wc_from_height)
    Wc = min(Wc, W)
    Hc = Wc / aspect
    if Hc > H:
        Hc = H
        Wc = Hc * aspect

    # --- 水平方向の配置 ---
    # 予測された左右の余白(L, R)はクロップの「幅」を決めるのに使い、
    # 「位置」は人物のバウンディングボックスの中心が画像の水平中央に来るよう配置する。
    body_cx = (x1 + x2) / 2.0

    # 注意: ここでクロップの幅(Wc)は変更しない。
    # 中央に寄せるために幅を詰めると、学習した「どこまで寄る/引くか」まで
    # 変わってしまい、手動クロップの再現性が落ちる。
    # 中央配置して画像外にはみ出す場合は、このあとの処理で画像内に収める
    # (その結果、人物は中央から多少ずれるが、クロップの大きさは保たれる)。
    if HORIZONTAL_CENTER_WEIGHT > 0.0:
        crop_left_centered = body_cx - Wc / 2.0
    else:
        crop_left_centered = None

    crop_left_asym = x1 - L * person_w
    if crop_left_centered is None:
        crop_left = crop_left_asym
    else:
        w = HORIZONTAL_CENTER_WEIGHT
        crop_left = w * crop_left_centered + (1.0 - w) * crop_left_asym

    crop_right = crop_left + Wc
    if crop_left < 0:
        crop_right -= crop_left
        crop_left = 0.0
    if crop_right > W:
        crop_left -= (crop_right - W)
        crop_right = W
    crop_left = max(0.0, crop_left)
    crop_right = min(float(W), crop_right)

    crop_top = y1 - T * person_h
    crop_bottom = crop_top + Hc
    if crop_top < 0:
        crop_bottom -= crop_top
        crop_top = 0.0
    if crop_bottom > H:
        crop_top -= (crop_bottom - H)
        crop_bottom = H
    crop_top = max(0.0, crop_top)
    crop_bottom = min(float(H), crop_bottom)

    return {
        "CropLeft": crop_left / W,
        "CropRight": crop_right / W,
        "CropTop": crop_top / H,
        "CropBottom": crop_bottom / H,
    }


def compute_crop_general(info, model_bundle=None):
    """汎用ポートレートモードのクロップ矩形(表示向き座標、0-1)を計算する。

    model_bundle があればそれによる予測、無ければヒューリスティックを使う。
    戻り値は (crop, features, margins) の3つ組(features/marginsはデバッグ・分析用)。
    """
    features = extract_features(info)
    if model_bundle is not None:
        margins = predict_margins_model(features, model_bundle)
    else:
        margins = heuristic_margins(features)
    return margins_to_crop_box(info, margins), features, margins
