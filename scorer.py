"""カップウィズハンドル スコアリングモジュール"""
from __future__ import annotations

from dataclasses import dataclass
from detector import DetectionResult


@dataclass
class ScoreBreakdown:
  """スコアの内訳"""
  cup_shape: float = 0.0       # カップ形状（U字度）: 最大20点
  cup_depth: float = 0.0       # カップの深さ: 最大15点
  rim_symmetry: float = 0.0    # 左右リムの対称性: 最大15点
  cup_period: float = 0.0      # カップの期間: 最大10点
  handle_depth: float = 0.0    # ハンドルの深さ: 最大15点
  handle_volume: float = 0.0   # ハンドルの出来高減少: 最大15点
  breakout_volume: float = 0.0 # ブレイクアウト出来高: 最大10点

  @property
  def total(self) -> float:
    return (
      self.cup_shape + self.cup_depth + self.rim_symmetry
      + self.cup_period + self.handle_depth + self.handle_volume
      + self.breakout_volume
    )


def calculate_score(result: DetectionResult) -> ScoreBreakdown:
  """
  検出結果からスコアを計算する（0〜100点）。

  スコア配分:
  - カップ形状（U字度）: 20点
  - カップの深さ（20〜35%が理想）: 15点
  - 左右リムの対称性: 15点
  - カップの期間（7〜15週が理想）: 10点
  - ハンドルの深さ（カップ上半分にあるほど高得点）: 15点
  - ハンドルの出来高減少: 15点
  - ブレイクアウト出来高: 10点
  """
  score = ScoreBreakdown()

  if not result.pattern_found:
    return score

  # 1. カップ形状（U字度）: 最大20点
  if result.u_shape_score is not None:
    score.cup_shape = result.u_shape_score * 20

  # 2. カップの深さ: 最大15点
  # 20〜35%が理想で15点、範囲外は減点
  if result.cup_depth is not None:
    depth = result.cup_depth
    if 0.20 <= depth <= 0.35:
      score.cup_depth = 15.0
    elif 0.15 <= depth < 0.20:
      # 15〜20%: 線形に減点
      score.cup_depth = 15.0 * (depth - 0.15) / 0.05
    elif 0.35 < depth <= 0.50:
      # 35〜50%: 線形に減点
      score.cup_depth = 15.0 * (0.50 - depth) / 0.15
    else:
      score.cup_depth = 0.0

  # 3. 左右リムの対称性: 最大15点
  if result.left_rim_price is not None and result.right_rim_price is not None:
    rim_diff = abs(result.left_rim_price - result.right_rim_price) / result.left_rim_price
    # 差が0%で15点、5%で0点
    score.rim_symmetry = max(0, 15.0 * (1 - rim_diff / 0.05))

  # 4. カップの期間: 最大10点
  # 7〜15週（49〜105日）が理想
  if result.cup_period is not None:
    period = result.cup_period
    if 49 <= period <= 105:
      score.cup_period = 10.0
    elif 30 <= period < 49:
      score.cup_period = 10.0 * (period - 30) / 19
    elif 105 < period <= 150:
      score.cup_period = 10.0 * (150 - period) / 45
    else:
      score.cup_period = 0.0

  # 5. ハンドルの深さ: 最大15点
  # カップの上半分にハンドル底があるほど高得点
  if (result.handle_bottom_price is not None
      and result.bottom_price is not None
      and result.left_rim_price is not None):
    cup_range = result.left_rim_price - result.bottom_price
    if cup_range > 0:
      # ハンドル底がカップのどの位置にあるか（0=底、1=リム）
      handle_position = (result.handle_bottom_price - result.bottom_price) / cup_range
      # 上半分（0.5〜1.0）にあるほど高得点
      if handle_position >= 0.5:
        score.handle_depth = 15.0
      else:
        score.handle_depth = 15.0 * handle_position / 0.5

  # 6. ハンドルの出来高減少: 最大15点
  # 30%以上減少で満点
  if result.handle_volume_decrease_ratio is not None:
    decrease = result.handle_volume_decrease_ratio
    if decrease >= 0.30:
      score.handle_volume = 15.0
    elif decrease > 0:
      score.handle_volume = 15.0 * decrease / 0.30
    else:
      score.handle_volume = 0.0

  # 7. ブレイクアウト出来高: 最大10点
  # 1.5倍以上で満点
  if result.breakout_volume_ratio is not None and result.breakout_idx is not None:
    vol_ratio = result.breakout_volume_ratio
    if vol_ratio >= 1.5:
      score.breakout_volume = 10.0
    elif vol_ratio >= 1.0:
      score.breakout_volume = 10.0 * (vol_ratio - 1.0) / 0.5
    else:
      score.breakout_volume = 0.0

  return score
