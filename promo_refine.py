# -*- coding: utf-8 -*-
"""
promo_refine.py

promoモードのクロップを、手動クロップ実績から学習したモデルで微調整する。

promoモードは固定ルールで既に手動クロップとよく一致する(実測 平均IoU 0.91)が、
半身写真や複数人写真では、ルールでは表現しきれない写真ごとの判断が残る。
そこを埋めるための「差分の学習」を行う。

【なぜ差分(residual)を学習するのか】
クロップ枠をゼロから予測するより、既に良い精度が出ている promo の結果からの
ズレだけを学習するほうが、予測すべき量が小さく精度を出しやすい。加えて、
モデルが外した場合も promo の結果から大きく離れないため安全側に働く。

【なぜ3つの値だけを予測するのか】
クロップ枠は元画像とアスペクト比が同じでなければならない。4辺を独立に予測すると
この制約が壊れるため、枠を一意に決められる最小の組
    中心の水平ズレ / 中心の垂直ズレ / 拡大率(対数)
の3つだけを予測する。こうすればどんな予測値でも必ず有効な枠になる。
予測後も crop_calculator の _clamp_keeping() を通すので、人物がクロップから
外れることはない。

【特徴量を増やす前に】
画像由来の特徴量(エッジ密度、セグメンテーションマスク)と、追加のポーズ特徴量
(腕の広がり、胴体の傾き、座り姿勢など)はいずれも検証済みで、交差検証での改善は
誤差の範囲(+0.002程度)にとどまり、枚数の少ない複数人写真ではむしろ悪化した。
精度を上げたい場合は、特徴量を足すのではなく手動クロップの枚数を増やすのが有効。

学習: train_promo_refine.py
使用: crop_calculator.py --mode promo --refine-model promo_refine_model.pkl
"""

import math
import os
import pickle

import numpy as np

import crop_calculator as cc

FEATURE_NAMES = [
    "full_body",          # 全身判定
    "n_persons",          # 検出人数
    "detect_stage",       # 人物検出のフォールバック段階(0=通常条件)
    "aspect",             # 画像のアスペクト比
    "person_w_ratio",     # 人物の幅 / 画像幅
    "person_h_ratio",     # 人物の高さ / 画像高さ
    "bbox_cx_ratio",      # 人物中心の水平位置
    "bbox_cy_ratio",      # 人物中心の垂直位置
    "gaze_dir",           # 視線・体の向き(-1=左, +1=右)
    "legs_cut",           # 脚がフレーム下端で切れているか
    "ankle_y",            # 足首の推定y座標(1.0超なら画面外)
    "ankle_vis",          # 足首の可視性
    "knee_vis",           # 膝の可視性
    "hip_vis",            # 腰の可視性
    "room_top",           # 人物の上に残っている余白 / 画像高さ
    "room_bottom",        # 人物の下に残っている余白
    "room_left",          # 人物の左に残っている余白
    "room_right",         # 人物の右に残っている余白
    "shoulder_w_ratio",   # 肩幅 / 人物幅
    "person_aspect",      # 人物の幅 / 高さ
]

# 予測値の安全クランプ(暴走した予測で極端なクロップにならないように)
MAX_SHIFT = 0.5        # 中心のズレ(人物サイズに対する比率)
MAX_LOG_SCALE = 0.5    # 拡大率の対数(±0.5 ≒ 0.61〜1.65倍)


def extract_features(info, poses=None, detect_stage=None):
    """クロップ結果の微調整に使う特徴量を作る。

    学習時(キャッシュしたランドマーク)と推論時(実際の検出結果)で同じ関数を
    使うことで、両者がずれないようにしている。
    """
    if poses is None:
        poses = info.get("poses") or [info["landmarks"]]
    if detect_stage is None:
        detect_stage = info.get("detect_stage") or 0

    W, H = info["W"], info["H"]
    x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]
    person_w = max(1e-6, x2 - x1)
    person_h = max(1e-6, y2 - y1)
    lm = poses[0]

    def vis(idx):
        return lm[idx].visibility

    ls, rs = lm[cc.LEFT_SHOULDER], lm[cc.RIGHT_SHOULDER]
    shoulder_w = abs(ls.x - rs.x) * W

    # 視線・体の向き: 肩の中点に対する鼻の水平方向のズレ
    gaze = 0.0
    if min(ls.visibility, rs.visibility) >= cc.VISIBILITY_THRESHOLD and shoulder_w > 1.0:
        mid = (ls.x + rs.x) / 2.0 * W
        ref = lm[cc.NOSE].x * W if vis(cc.NOSE) >= cc.VISIBILITY_THRESHOLD else mid
        gaze = float(np.clip((ref - mid) / (shoulder_w / 2.0), -1.0, 1.0))

    # 足首の推定位置。1.0を超えていれば脚がフレームの下で切れている
    ankle_y = max(lm[cc.LEFT_ANKLE].y, lm[cc.RIGHT_ANKLE].y)

    return {
        "full_body": 1.0 if info["full_body"] else 0.0,
        "n_persons": float(info.get("n_persons", 1)),
        "detect_stage": float(detect_stage),
        "aspect": W / H,
        "person_w_ratio": person_w / W,
        "person_h_ratio": person_h / H,
        "bbox_cx_ratio": ((x1 + x2) / 2.0) / W,
        "bbox_cy_ratio": ((y1 + y2) / 2.0) / H,
        "gaze_dir": gaze,
        "legs_cut": 1.0 if ankle_y > 1.0 else 0.0,
        "ankle_y": float(min(ankle_y, 3.0)),
        "ankle_vis": max(vis(cc.LEFT_ANKLE), vis(cc.RIGHT_ANKLE)),
        "knee_vis": max(vis(cc.LEFT_KNEE), vis(cc.RIGHT_KNEE)),
        "hip_vis": max(vis(cc.LEFT_HIP), vis(cc.RIGHT_HIP)),
        "room_top": y1 / H,
        "room_bottom": (H - y2) / H,
        "room_left": x1 / W,
        "room_right": (W - x2) / W,
        "shoulder_w_ratio": shoulder_w / person_w,
        "person_aspect": person_w / person_h,
    }


def features_to_vector(features):
    return np.array([[features[name] for name in FEATURE_NAMES]])


def crop_target(base_crop, hand_crop, info):
    """promoのクロップと手動クロップの差を、学習の目標値に変換する。

    戻り値は (中心の水平ズレ, 中心の垂直ズレ, 拡大率の対数)。
    """
    W, H = info["W"], info["H"]
    person_w = max(1e-6, info["x2"] - info["x1"])
    person_h = max(1e-6, info["y2"] - info["y1"])

    bw = (base_crop["CropRight"] - base_crop["CropLeft"]) * W
    hw = (hand_crop["CropRight"] - hand_crop["CropLeft"]) * W
    if bw <= 0 or hw <= 0:
        return None

    base_cx = (base_crop["CropLeft"] + base_crop["CropRight"]) / 2.0 * W
    base_cy = (base_crop["CropTop"] + base_crop["CropBottom"]) / 2.0 * H
    hand_cx = (hand_crop["CropLeft"] + hand_crop["CropRight"]) / 2.0 * W
    hand_cy = (hand_crop["CropTop"] + hand_crop["CropBottom"]) / 2.0 * H

    return [
        (hand_cx - base_cx) / person_w,
        (hand_cy - base_cy) / person_h,
        math.log(hw / bw),
    ]


def apply_delta(base_crop, delta, info):
    """予測された差分をpromoのクロップに適用する。

    アスペクト比は元のクロップ(=元画像)と同じに保ち、画像からはみ出さず、
    人物がクロップから外れないように調整する。
    """
    W, H = info["W"], info["H"]
    person_w = max(1e-6, info["x2"] - info["x1"])
    person_h = max(1e-6, info["y2"] - info["y1"])

    left = base_crop["CropLeft"] * W
    right = base_crop["CropRight"] * W
    top = base_crop["CropTop"] * H
    bottom = base_crop["CropBottom"] * H
    bw, bh = right - left, bottom - top
    if bw <= 0 or bh <= 0:
        return base_crop

    d_cx = float(np.clip(delta[0], -MAX_SHIFT, MAX_SHIFT))
    d_cy = float(np.clip(delta[1], -MAX_SHIFT, MAX_SHIFT))
    d_scale = float(np.clip(delta[2], -MAX_LOG_SCALE, MAX_LOG_SCALE))

    new_w = min(bw * math.exp(d_scale), float(W))
    new_h = new_w * bh / bw
    if new_h > H:
        new_h = float(H)
        new_w = new_h * bw / bh

    cx = (left + right) / 2.0 + d_cx * person_w
    cy = (top + bottom) / 2.0 + d_cy * person_h

    new_left = cc._clamp_keeping(cx - new_w / 2.0, W - new_w, info["x2"] - new_w, info["x1"])
    new_top = cc._clamp_keeping(cy - new_h / 2.0, H - new_h, info["y2"] - new_h, info["y1"])

    return {
        "CropLeft": new_left / W,
        "CropRight": (new_left + new_w) / W,
        "CropTop": new_top / H,
        "CropBottom": (new_top + new_h) / H,
    }


def load_model(path):
    """学習済みモデルを読み込む。ファイルが無ければ None を返す。"""
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def refine(base_crop, info, model_bundle):
    """promoのクロップをモデルで微調整する。モデルが無ければそのまま返す。"""
    if model_bundle is None:
        return base_crop
    features = extract_features(info)
    x = np.array([[features[name] for name in model_bundle["feature_names"]]])
    delta = [float(model_bundle[k].predict(x)[0]) for k in ("d_cx", "d_cy", "d_scale")]
    return apply_delta(base_crop, delta, info)
