"""カップウィズハンドル パターン検出エンジン"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


@dataclass
class DetectionParams:
  """検出パラメータ"""
  # カップの深さ範囲（左リムからの下落率）
  cup_depth_min: float = 0.15
  cup_depth_max: float = 0.50
  # カップ期間（日数）
  cup_period_min: int = 30
  cup_period_max: int = 150
  # 左右リムの許容誤差
  rim_tolerance: float = 0.05
  # ハンドルの最大下落率
  handle_depth_max: float = 0.15
  # ハンドル期間（日数）
  handle_period_min: int = 5
  handle_period_max: int = 25
  # ハンドル出来高減少率
  handle_volume_decrease: float = 0.30
  # ブレイクアウト出来高倍率
  breakout_volume_ratio: float = 1.5


@dataclass
class StepResult:
  """各ステップの検出結果"""
  step_num: int
  name: str
  detected: bool
  detail: str
  value: float | None = None


@dataclass
class DetectionResult:
  """パターン検出結果"""
  # パターン検出されたか
  pattern_found: bool = False
  # ステータス: "ブレイクアウト済み", "ブレイクアウト待ち", "ハンドル形成中", "パターンなし"
  status: str = "パターンなし"
  # 各ステップの結果
  steps: list[StepResult] = field(default_factory=list)
  # キーポイントのインデックス
  left_rim_idx: int | None = None
  bottom_idx: int | None = None
  right_rim_idx: int | None = None
  handle_bottom_idx: int | None = None
  breakout_idx: int | None = None
  # 値
  left_rim_price: float | None = None
  bottom_price: float | None = None
  right_rim_price: float | None = None
  handle_bottom_price: float | None = None
  # カップの深さ（下落率）
  cup_depth: float | None = None
  # カップ期間（日数）
  cup_period: int | None = None
  # ハンドルの深さ（下落率）
  handle_depth: float | None = None
  # ハンドル期間（日数）
  handle_period: int | None = None
  # U字スコア（0〜1、1が最もU字）
  u_shape_score: float | None = None
  # ハンドル出来高減少率
  handle_volume_decrease_ratio: float | None = None
  # ブレイクアウト出来高倍率
  breakout_volume_ratio: float | None = None
  # ブレイクアウト目標価格
  target_price: float | None = None


def detect_cup_with_handle(
  df: pd.DataFrame,
  params: DetectionParams | None = None,
) -> DetectionResult:
  """
  カップウィズハンドルパターンを検出する。

  Args:
    df: 株価データ（open, high, low, close, volume カラム必須）
    params: 検出パラメータ（Noneの場合はデフォルト値）

  Returns:
    DetectionResult: 検出結果
  """
  if params is None:
    params = DetectionParams()

  result = DetectionResult()
  close = df["close"].values
  volume = df["volume"].values
  n = len(close)

  if n < params.cup_period_min + params.handle_period_min:
    result.steps.append(StepResult(
      step_num=1, name="左リム検出",
      detected=False, detail="データが不足しています"
    ))
    return result

  # ステップ1: 左リム検出
  # 局所高値を検出（前後10日間で最大の点）
  order = min(10, n // 5)
  if order < 2:
    order = 2
  local_maxima = argrelextrema(close, np.greater_equal, order=order)[0]

  if len(local_maxima) == 0:
    result.steps.append(StepResult(
      step_num=1, name="左リム検出",
      detected=False, detail="局所高値が見つかりません"
    ))
    return result

  # 最良のカップウィズハンドルパターンを探索
  best_result = None
  best_score = -1

  for left_rim_candidate in local_maxima:
    # 左リムが後半すぎる場合はスキップ（カップ+ハンドルの余地が必要）
    min_remaining = params.cup_period_min + params.handle_period_min
    if left_rim_candidate > n - min_remaining:
      continue

    candidate = _try_detect_from_left_rim(
      close, volume, left_rim_candidate, params, n
    )
    if candidate is not None and candidate["score"] > best_score:
      best_result = candidate
      best_score = candidate["score"]

  if best_result is None:
    # パターンが見つからなかった場合、部分的な検出を試みる
    result = _detect_partial_pattern(close, volume, local_maxima, params, n)
    return result

  # 最良の結果を DetectionResult に変換
  result = _build_result(best_result, close, volume, params)
  return result


def _try_detect_from_left_rim(
  close: np.ndarray,
  volume: np.ndarray,
  left_rim_idx: int,
  params: DetectionParams,
  n: int,
) -> dict | None:
  """特定の左リムからカップウィズハンドルの検出を試みる"""
  left_rim_price = close[left_rim_idx]

  # ステップ2: 下落フェーズ検出
  # 左リム以降で最安値を探す
  search_end = min(left_rim_idx + params.cup_period_max, n)
  if search_end <= left_rim_idx + 10:
    return None

  segment = close[left_rim_idx:search_end]
  bottom_offset = np.argmin(segment)
  bottom_idx = left_rim_idx + bottom_offset
  bottom_price = close[bottom_idx]

  # 下落率の確認
  drop_rate = (left_rim_price - bottom_price) / left_rim_price
  if drop_rate < params.cup_depth_min or drop_rate > params.cup_depth_max:
    return None

  # ステップ3: 底のU字型判定
  # 底付近（±3%）の滞在日数を計算
  bottom_zone = bottom_price * 1.03
  bottom_range = close[left_rim_idx:search_end]
  days_near_bottom = np.sum(bottom_range <= bottom_zone)
  cup_width = search_end - left_rim_idx
  # U字度: 底付近の滞在比率が高いほどU字
  u_shape_ratio = days_near_bottom / max(cup_width, 1)
  # V字の場合は滞在日数が少ない → u_shape_ratioが低い
  u_shape_score = min(u_shape_ratio * 5, 1.0)  # 20%以上滞在でスコア1.0

  # ステップ4&5: 回復フェーズ＆右リム検出
  # 底以降で左リムと同水準まで回復した点を探す
  right_rim_idx = None
  right_rim_price = None

  for i in range(bottom_idx + 5, search_end):
    price = close[i]
    recovery_rate = abs(price - left_rim_price) / left_rim_price
    if recovery_rate <= params.rim_tolerance:
      right_rim_idx = i
      right_rim_price = price
      break

  if right_rim_idx is None:
    return None

  # カップ期間の確認
  cup_period = right_rim_idx - left_rim_idx
  if cup_period < params.cup_period_min or cup_period > params.cup_period_max:
    return None

  # 回復フェーズの出来高増加確認
  mid_point = (bottom_idx + right_rim_idx) // 2
  vol_first_half = np.mean(volume[bottom_idx:mid_point]) if mid_point > bottom_idx else 0
  vol_second_half = np.mean(volume[mid_point:right_rim_idx]) if right_rim_idx > mid_point else 0
  recovery_volume_increasing = vol_second_half > vol_first_half * 0.8

  # ステップ6: ハンドル検出
  handle_start = right_rim_idx
  handle_search_end = min(handle_start + params.handle_period_max + 5, n)

  if handle_search_end <= handle_start + params.handle_period_min:
    return None

  handle_segment = close[handle_start:handle_search_end]
  handle_bottom_offset = np.argmin(handle_segment)
  handle_bottom_idx = handle_start + handle_bottom_offset
  handle_bottom_price = close[handle_bottom_idx]

  # ハンドルの下落率
  handle_depth = (right_rim_price - handle_bottom_price) / right_rim_price
  if handle_depth > params.handle_depth_max or handle_depth < 0.01:
    return None

  # ハンドル期間
  handle_period = handle_bottom_idx - handle_start
  if handle_period < 1:
    return None

  # ハンドル出来高がカップ部分より減少しているか
  cup_avg_volume = np.mean(volume[left_rim_idx:right_rim_idx])
  handle_avg_volume = np.mean(volume[handle_start:handle_search_end])
  handle_vol_decrease = 1 - (handle_avg_volume / max(cup_avg_volume, 1))

  # ステップ7: ハンドル回復検出
  handle_recovering = False
  if handle_bottom_idx < n - 1:
    post_handle = close[handle_bottom_idx:min(handle_bottom_idx + 10, n)]
    if len(post_handle) > 1 and post_handle[-1] > post_handle[0]:
      handle_recovering = True

  # ステップ8: ブレイクアウト判定
  breakout_idx = None
  breakout_vol_ratio = 0.0
  breakout_level = max(left_rim_price, right_rim_price)
  avg_volume_20 = np.mean(volume[max(0, handle_bottom_idx - 20):handle_bottom_idx]) if handle_bottom_idx > 20 else np.mean(volume[:handle_bottom_idx])

  for i in range(handle_bottom_idx, min(handle_bottom_idx + 15, n)):
    if close[i] > breakout_level:
      breakout_idx = i
      if avg_volume_20 > 0:
        breakout_vol_ratio = volume[i] / avg_volume_20
      break

  # スコア計算（簡易版、後でscorer.pyが正式に計算）
  score = 0
  score += u_shape_score * 20
  if 0.20 <= drop_rate <= 0.35:
    score += 15
  elif 0.15 <= drop_rate <= 0.50:
    score += 8
  rim_diff = abs(left_rim_price - right_rim_price) / left_rim_price
  score += max(0, 15 - rim_diff * 150)
  if breakout_idx is not None:
    score += 10

  return {
    "score": score,
    "left_rim_idx": left_rim_idx,
    "left_rim_price": left_rim_price,
    "bottom_idx": bottom_idx,
    "bottom_price": bottom_price,
    "right_rim_idx": right_rim_idx,
    "right_rim_price": right_rim_price,
    "handle_bottom_idx": handle_bottom_idx,
    "handle_bottom_price": handle_bottom_price,
    "breakout_idx": breakout_idx,
    "cup_depth": drop_rate,
    "cup_period": cup_period,
    "handle_depth": handle_depth,
    "handle_period": handle_period,
    "u_shape_score": u_shape_score,
    "handle_vol_decrease": handle_vol_decrease,
    "breakout_vol_ratio": breakout_vol_ratio,
    "recovery_volume_increasing": recovery_volume_increasing,
    "handle_recovering": handle_recovering,
  }


def _build_result(
  data: dict,
  close: np.ndarray,
  volume: np.ndarray,
  params: DetectionParams,
) -> DetectionResult:
  """検出データからDetectionResultを構築する"""
  result = DetectionResult()
  result.pattern_found = True
  result.left_rim_idx = data["left_rim_idx"]
  result.left_rim_price = data["left_rim_price"]
  result.bottom_idx = data["bottom_idx"]
  result.bottom_price = data["bottom_price"]
  result.right_rim_idx = data["right_rim_idx"]
  result.right_rim_price = data["right_rim_price"]
  result.handle_bottom_idx = data["handle_bottom_idx"]
  result.handle_bottom_price = data["handle_bottom_price"]
  result.breakout_idx = data["breakout_idx"]
  result.cup_depth = data["cup_depth"]
  result.cup_period = data["cup_period"]
  result.handle_depth = data["handle_depth"]
  result.handle_period = data["handle_period"]
  result.u_shape_score = data["u_shape_score"]
  result.handle_volume_decrease_ratio = data["handle_vol_decrease"]
  result.breakout_volume_ratio = data["breakout_vol_ratio"]

  # ブレイクアウト目標価格（カップの深さ分を右リムに加算）
  cup_height = data["left_rim_price"] - data["bottom_price"]
  result.target_price = max(data["left_rim_price"], data["right_rim_price"]) + cup_height

  # ステータス判定
  if data["breakout_idx"] is not None:
    if data["breakout_vol_ratio"] >= params.breakout_volume_ratio:
      result.status = "ブレイクアウト済み"
    else:
      result.status = "ブレイクアウト済み（出来高不足）"
  elif data["handle_recovering"]:
    result.status = "ブレイクアウト待ち"
  else:
    result.status = "ハンドル形成中"

  # 各ステップの結果を構築
  result.steps = [
    StepResult(
      step_num=1, name="左リム検出", detected=True,
      detail=f"価格: {data['left_rim_price']:.2f}",
      value=data["left_rim_price"],
    ),
    StepResult(
      step_num=2, name="下落フェーズ検出", detected=True,
      detail=f"下落率: {data['cup_depth']:.1%}",
      value=data["cup_depth"],
    ),
    StepResult(
      step_num=3, name="底の検出（U字型判定）", detected=True,
      detail=f"底値: {data['bottom_price']:.2f} / U字スコア: {data['u_shape_score']:.2f}",
      value=data["u_shape_score"],
    ),
    StepResult(
      step_num=4, name="回復フェーズ検出",
      detected=data["recovery_volume_increasing"],
      detail="出来高増加あり" if data["recovery_volume_increasing"] else "出来高増加なし",
    ),
    StepResult(
      step_num=5, name="右リム検出", detected=True,
      detail=f"価格: {data['right_rim_price']:.2f} / カップ期間: {data['cup_period']}日",
      value=data["right_rim_price"],
    ),
    StepResult(
      step_num=6, name="ハンドル検出", detected=True,
      detail=f"下落率: {data['handle_depth']:.1%} / 出来高減少: {data['handle_vol_decrease']:.1%}",
      value=data["handle_depth"],
    ),
    StepResult(
      step_num=7, name="ハンドル回復検出",
      detected=data["handle_recovering"],
      detail="回復トレンドあり" if data["handle_recovering"] else "回復トレンドなし",
    ),
    StepResult(
      step_num=8, name="ブレイクアウト判定",
      detected=data["breakout_idx"] is not None,
      detail=(
        f"出来高倍率: {data['breakout_vol_ratio']:.2f}x"
        if data["breakout_idx"] is not None
        else "未ブレイクアウト"
      ),
      value=data["breakout_vol_ratio"] if data["breakout_idx"] is not None else None,
    ),
  ]

  return result


def _detect_partial_pattern(
  close: np.ndarray,
  volume: np.ndarray,
  local_maxima: np.ndarray,
  params: DetectionParams,
  n: int,
) -> DetectionResult:
  """部分的なパターン検出（形成中の可能性を判定）"""
  result = DetectionResult()
  steps = []

  # 最も高い局所高値を左リム候補とする
  best_left_idx = local_maxima[np.argmax(close[local_maxima])]
  left_price = close[best_left_idx]

  steps.append(StepResult(
    step_num=1, name="左リム検出", detected=True,
    detail=f"価格: {left_price:.2f}",
    value=left_price,
  ))

  # 下落フェーズの確認
  if best_left_idx < n - 10:
    after_left = close[best_left_idx:]
    bottom_offset = np.argmin(after_left)
    bottom_idx = best_left_idx + bottom_offset
    bottom_price = close[bottom_idx]
    drop_rate = (left_price - bottom_price) / left_price

    step2_detected = params.cup_depth_min <= drop_rate <= params.cup_depth_max
    steps.append(StepResult(
      step_num=2, name="下落フェーズ検出",
      detected=step2_detected,
      detail=f"下落率: {drop_rate:.1%}" + ("" if step2_detected else " （範囲外）"),
      value=drop_rate,
    ))

    if step2_detected:
      result.left_rim_idx = best_left_idx
      result.left_rim_price = left_price
      result.bottom_idx = bottom_idx
      result.bottom_price = bottom_price
      result.cup_depth = drop_rate

      # 底のU字型判定
      bottom_zone = bottom_price * 1.03
      days_near = np.sum(after_left <= bottom_zone)
      u_score = min((days_near / max(len(after_left), 1)) * 5, 1.0)
      steps.append(StepResult(
        step_num=3, name="底の検出（U字型判定）", detected=True,
        detail=f"底値: {bottom_price:.2f} / U字スコア: {u_score:.2f}",
        value=u_score,
      ))
      result.u_shape_score = u_score

      # 現在値が回復中か確認
      current_price = close[-1]
      recovery_rate = (current_price - bottom_price) / (left_price - bottom_price) if left_price > bottom_price else 0
      steps.append(StepResult(
        step_num=4, name="回復フェーズ検出",
        detected=recovery_rate > 0.3,
        detail=f"回復率: {recovery_rate:.1%}",
        value=recovery_rate,
      ))

      # 右リムに達しているか
      rim_reached = abs(current_price - left_price) / left_price <= params.rim_tolerance
      if rim_reached:
        steps.append(StepResult(
          step_num=5, name="右リム検出", detected=True,
          detail=f"現在値が左リム水準に到達",
        ))
        result.status = "ハンドル形成中"
      else:
        steps.append(StepResult(
          step_num=5, name="右リム検出", detected=False,
          detail=f"現在値: {current_price:.2f} / 目標: {left_price:.2f}付近",
        ))
        if recovery_rate > 0.5:
          result.status = "パターンなし（回復途中）"
    else:
      steps.append(StepResult(
        step_num=3, name="底の検出（U字型判定）",
        detected=False, detail="下落フェーズ未検出のためスキップ",
      ))
  else:
    steps.append(StepResult(
      step_num=2, name="下落フェーズ検出",
      detected=False, detail="データ不足",
    ))

  # 未検出ステップを埋める
  existing_steps = {s.step_num for s in steps}
  for i in range(1, 9):
    if i not in existing_steps:
      step_names = {
        1: "左リム検出", 2: "下落フェーズ検出",
        3: "底の検出（U字型判定）", 4: "回復フェーズ検出",
        5: "右リム検出", 6: "ハンドル検出",
        7: "ハンドル回復検出", 8: "ブレイクアウト判定",
      }
      steps.append(StepResult(
        step_num=i, name=step_names[i],
        detected=False, detail="前ステップ未検出のためスキップ",
      ))

  steps.sort(key=lambda s: s.step_num)
  result.steps = steps
  return result
