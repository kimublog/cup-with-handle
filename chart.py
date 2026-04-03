"""plotlyチャート生成モジュール"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from detector import DetectionResult


def create_chart(
  df: pd.DataFrame,
  result: DetectionResult,
  ticker: str = "",
  stock_name: str = "",
) -> go.Figure:
  """
  ローソク足チャート＋出来高＋パターンハイライトを生成する。

  Args:
    df: 株価データ
    result: パターン検出結果
    ticker: 銘柄コード
    stock_name: 銘柄名

  Returns:
    plotly Figure
  """
  # サブプロット（上: ローソク足、下: 出来高）
  fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.75, 0.25],
    subplot_titles=("", "出来高"),
  )

  dates = df.index

  # ローソク足
  fig.add_trace(
    go.Candlestick(
      x=dates,
      open=df["open"],
      high=df["high"],
      low=df["low"],
      close=df["close"],
      name="価格",
      increasing_line_color="#26a69a",
      decreasing_line_color="#ef5350",
    ),
    row=1, col=1,
  )

  # 出来高バーチャート
  colors = [
    "#26a69a" if df["close"].iloc[i] >= df["open"].iloc[i] else "#ef5350"
    for i in range(len(df))
  ]
  fig.add_trace(
    go.Bar(
      x=dates,
      y=df["volume"],
      name="出来高",
      marker_color=colors,
      opacity=0.7,
    ),
    row=2, col=1,
  )

  # パターンが検出された場合、ハイライトとマーカーを追加
  if result.pattern_found:
    _add_pattern_highlights(fig, df, result, dates)
    _add_markers(fig, df, result, dates)

  # レイアウト設定
  title = f"{stock_name} ({ticker})" if stock_name else ticker
  fig.update_layout(
    title=dict(text=title, font=dict(size=18)),
    xaxis_rangeslider_visible=False,
    height=650,
    template="plotly_white",
    legend=dict(
      orientation="h",
      yanchor="bottom",
      y=1.02,
      xanchor="right",
      x=1,
    ),
    margin=dict(l=60, r=30, t=80, b=40),
  )

  fig.update_yaxes(title_text="価格", row=1, col=1)
  fig.update_yaxes(title_text="出来高", row=2, col=1)

  return fig


def _add_pattern_highlights(
  fig: go.Figure,
  df: pd.DataFrame,
  result: DetectionResult,
  dates: pd.DatetimeIndex,
) -> None:
  """カップ・ハンドル部分を色付き背景でハイライト"""
  # カップ部分のハイライト（青色半透明）
  if result.left_rim_idx is not None and result.right_rim_idx is not None:
    cup_start = dates[result.left_rim_idx]
    cup_end = dates[result.right_rim_idx]
    y_min = df["low"].iloc[result.left_rim_idx:result.right_rim_idx + 1].min()
    y_max = df["high"].iloc[result.left_rim_idx:result.right_rim_idx + 1].max()

    fig.add_shape(
      type="rect",
      x0=cup_start, x1=cup_end,
      y0=y_min * 0.99, y1=y_max * 1.01,
      fillcolor="rgba(66, 133, 244, 0.1)",
      line=dict(color="rgba(66, 133, 244, 0.3)", width=1),
      row=1, col=1,
    )
    fig.add_annotation(
      x=dates[(result.left_rim_idx + result.right_rim_idx) // 2],
      y=y_max * 1.02,
      text="カップ",
      showarrow=False,
      font=dict(size=12, color="#4285F4"),
      row=1, col=1,
    )

  # ハンドル部分のハイライト（オレンジ色半透明）
  if result.right_rim_idx is not None and result.handle_bottom_idx is not None:
    handle_start = dates[result.right_rim_idx]
    handle_end_idx = min(
      result.handle_bottom_idx + 10,
      len(dates) - 1,
    )
    if result.breakout_idx is not None:
      handle_end_idx = result.breakout_idx
    handle_end = dates[handle_end_idx]

    h_slice = slice(result.right_rim_idx, handle_end_idx + 1)
    h_y_min = df["low"].iloc[h_slice].min()
    h_y_max = df["high"].iloc[h_slice].max()

    fig.add_shape(
      type="rect",
      x0=handle_start, x1=handle_end,
      y0=h_y_min * 0.99, y1=h_y_max * 1.01,
      fillcolor="rgba(255, 152, 0, 0.1)",
      line=dict(color="rgba(255, 152, 0, 0.3)", width=1),
      row=1, col=1,
    )
    fig.add_annotation(
      x=dates[(result.right_rim_idx + handle_end_idx) // 2],
      y=h_y_max * 1.02,
      text="ハンドル",
      showarrow=False,
      font=dict(size=12, color="#FF9800"),
      row=1, col=1,
    )


def _add_markers(
  fig: go.Figure,
  df: pd.DataFrame,
  result: DetectionResult,
  dates: pd.DatetimeIndex,
) -> None:
  """キーポイントにマーカーを追加"""
  markers = []

  if result.left_rim_idx is not None:
    markers.append({
      "idx": result.left_rim_idx,
      "label": "左リム",
      "color": "#4285F4",
      "symbol": "triangle-up",
    })

  if result.bottom_idx is not None:
    markers.append({
      "idx": result.bottom_idx,
      "label": "底",
      "color": "#EA4335",
      "symbol": "triangle-down",
    })

  if result.right_rim_idx is not None:
    markers.append({
      "idx": result.right_rim_idx,
      "label": "右リム",
      "color": "#4285F4",
      "symbol": "triangle-up",
    })

  if result.handle_bottom_idx is not None:
    markers.append({
      "idx": result.handle_bottom_idx,
      "label": "ハンドル底",
      "color": "#FF9800",
      "symbol": "triangle-down",
    })

  if result.breakout_idx is not None:
    markers.append({
      "idx": result.breakout_idx,
      "label": "ブレイクアウト",
      "color": "#34A853",
      "symbol": "star",
    })

  for m in markers:
    idx = m["idx"]
    if idx >= len(dates):
      continue
    fig.add_trace(
      go.Scatter(
        x=[dates[idx]],
        y=[df["close"].iloc[idx]],
        mode="markers+text",
        marker=dict(
          size=12,
          color=m["color"],
          symbol=m["symbol"],
          line=dict(width=1, color="white"),
        ),
        text=[m["label"]],
        textposition="top center",
        textfont=dict(size=10, color=m["color"]),
        name=m["label"],
        showlegend=False,
      ),
      row=1, col=1,
    )

  # ブレイクアウトラインを追加
  if result.left_rim_price is not None and result.right_rim_price is not None:
    breakout_level = max(result.left_rim_price, result.right_rim_price)
    start_idx = result.left_rim_idx if result.left_rim_idx is not None else 0
    fig.add_hline(
      y=breakout_level,
      line_dash="dash",
      line_color="rgba(52, 168, 83, 0.5)",
      annotation_text=f"ブレイクアウトライン: {breakout_level:.2f}",
      annotation_position="top left",
      annotation_font_color="#34A853",
      row=1, col=1,
    )

  # 目標価格ラインを追加
  if result.target_price is not None and result.breakout_idx is not None:
    fig.add_hline(
      y=result.target_price,
      line_dash="dot",
      line_color="rgba(156, 39, 176, 0.5)",
      annotation_text=f"目標価格: {result.target_price:.2f}",
      annotation_position="top left",
      annotation_font_color="#9C27B0",
      row=1, col=1,
    )
