"""カップウィズハンドル検出ツール — Streamlit メインアプリ"""

import streamlit as st
import pandas as pd
from data_fetcher import fetch_stock_data, get_stock_name
from detector import detect_cup_with_handle, DetectionParams
from scorer import calculate_score
from chart import create_chart
from stock_lists import get_stock_list, format_ticker, fetch_all_jpx_stocks
from data_fetcher import run_pre_filter

# ページ設定
st.set_page_config(
  page_title="カップウィズハンドル検出ツール",
  page_icon="📈",
  layout="wide",
)


def render_single_mode(market: str, period_months: int, params: DetectionParams):
  """単一銘柄分析モード"""
  placeholder = "例: 7203（トヨタ）" if market == "日本株" else "例: AAPL"
  ticker_input = st.text_input("銘柄コードを入力", placeholder=placeholder)

  if not ticker_input:
    st.info("銘柄コードを入力して分析を開始してください。")
    return

  ticker = format_ticker(ticker_input, market)

  with st.spinner(f"{ticker} のデータを取得中..."):
    df = fetch_stock_data(ticker, period_months)

  if df is None or df.empty:
    st.error(f"銘柄コード「{ticker}」のデータを取得できませんでした。銘柄コードを確認してください。")
    return

  stock_name = get_stock_name(ticker)

  # パターン検出
  with st.spinner("パターンを分析中..."):
    result = detect_cup_with_handle(df, params)
    score = calculate_score(result)

  # --- 結果表示 ---
  # ステータス表示
  status_colors = {
    "ブレイクアウト済み": "🟢",
    "ブレイクアウト済み（出来高不足）": "🟡",
    "ブレイクアウト待ち": "🟡",
    "ハンドル形成中": "🟠",
    "パターンなし": "🔴",
    "パターンなし（回復途中）": "🔴",
  }
  status_icon = status_colors.get(result.status, "⚪")

  col1, col2, col3 = st.columns(3)
  with col1:
    st.metric("判定結果", f"{status_icon} {result.status}")
  with col2:
    score_total = round(score.total, 1)
    st.metric("総合スコア", f"{score_total} / 100")
  with col3:
    current_price = df["close"].iloc[-1]
    st.metric("現在価格", f"{current_price:,.2f}")

  # チャート表示
  fig = create_chart(df, result, ticker, stock_name)
  st.plotly_chart(fig, use_container_width=True)

  # スコア内訳
  st.subheader("スコア内訳")
  score_data = {
    "項目": [
      "カップ形状（U字度）",
      "カップの深さ",
      "左右リムの対称性",
      "カップの期間",
      "ハンドルの深さ",
      "ハンドルの出来高減少",
      "ブレイクアウト出来高",
    ],
    "スコア": [
      f"{score.cup_shape:.1f}",
      f"{score.cup_depth:.1f}",
      f"{score.rim_symmetry:.1f}",
      f"{score.cup_period:.1f}",
      f"{score.handle_depth:.1f}",
      f"{score.handle_volume:.1f}",
      f"{score.breakout_volume:.1f}",
    ],
    "配点": ["20", "15", "15", "10", "15", "15", "10"],
  }
  st.dataframe(
    pd.DataFrame(score_data),
    hide_index=True,
    use_container_width=True,
  )

  # 各ステップの検出結果
  st.subheader("検出ステップ詳細")
  step_data = {
    "ステップ": [
      "①②③④⑤⑥⑦⑧"[s.step_num - 1] + " " + s.name
      if 1 <= s.step_num <= 8 else f"{s.step_num}. {s.name}"
      for s in result.steps
    ],
    "検出": ["✅" if s.detected else "❌" for s in result.steps],
    "詳細": [s.detail for s in result.steps],
  }
  st.dataframe(
    pd.DataFrame(step_data),
    hide_index=True,
    use_container_width=True,
  )

  # 目標価格
  if result.target_price is not None and result.pattern_found:
    st.info(f"ブレイクアウト目標価格: **{result.target_price:,.2f}**")


def render_screening_mode(market: str, period_months: int, params: DetectionParams):
  """一括スクリーニングモード"""
  stock_list = get_stock_list(market)
  list_name = "日経225主要銘柄" if market == "日本株" else "S&P500主要銘柄"

  # スキャン対象の選択肢
  scan_options = [f"{list_name}（{len(stock_list)}銘柄）", "カスタム入力"]
  if market == "日本株":
    scan_options.insert(1, "全上場企業（事前フィルタあり）")

  input_mode = st.radio(
    "スキャン対象",
    scan_options,
    horizontal=True,
  )

  use_full_scan = input_mode.startswith("全上場企業")

  if input_mode.startswith("カスタム"):
    custom_input = st.text_area(
      "銘柄コードをカンマ区切りで入力",
      placeholder="7203, 6758, 9984" if market == "日本株" else "AAPL, MSFT, GOOGL",
    )
    if custom_input:
      tickers = [format_ticker(t.strip(), market) for t in custom_input.split(",") if t.strip()]
      stock_list = {t: t for t in tickers}
    else:
      stock_list = {}
  elif use_full_scan:
    stock_list = None  # スキャン開始時に取得

  if stock_list is not None and not stock_list:
    st.info("スキャン対象の銘柄を入力してください。")
    return

  if use_full_scan:
    st.info(
      "東証上場の全銘柄（約3,700社）を対象にスキャンします。\n\n"
      "**処理の流れ:**\n"
      "1. JPXから銘柄一覧を取得\n"
      "2. 事前フィルタで候補を絞り込み（約5〜10分）\n"
      "3. 候補銘柄の詳細パターン検出\n"
    )

  if st.button("スキャン開始", type="primary"):
    # --- 全上場企業モード: 事前フィルタ付き ---
    if use_full_scan:
      _run_full_scan(market, period_months, params)
    else:
      _run_normal_scan(stock_list, period_months, params)


def _run_full_scan(market: str, period_months: int, params: DetectionParams):
  """全上場企業スキャン（事前フィルタ付き）"""
  # ステップ1: JPXから銘柄一覧取得
  with st.spinner("JPXから上場銘柄一覧を取得中..."):
    all_stocks = fetch_all_jpx_stocks()

  if not all_stocks:
    st.error("JPXから銘柄一覧を取得できませんでした。ネットワーク接続を確認してください。")
    return

  st.info(f"取得完了: **{len(all_stocks)}銘柄**")

  # ステップ2: 事前フィルタ
  st.subheader("ステップ1: 事前フィルタリング")
  filter_status = st.empty()
  filter_progress = st.progress(0, text="事前フィルタ中...")
  passed_count_display = st.empty()

  passed_tickers = {}
  filter_done = False

  def on_filter_progress(current, total, ticker):
    nonlocal filter_done
    ratio = current / total
    filter_progress.progress(
      ratio,
      text=f"事前フィルタ中... {ticker} ({current}/{total})",
    )
    if current == total:
      filter_done = True

  passed_tickers = run_pre_filter(
    all_stocks,
    period_months=period_months,
    max_workers=8,
    progress_callback=on_filter_progress,
  )

  filter_progress.empty()

  if not passed_tickers:
    st.warning("事前フィルタを通過した銘柄がありませんでした。")
    return

  filter_status.success(
    f"事前フィルタ完了: **{len(all_stocks)}銘柄** → **{len(passed_tickers)}銘柄** に絞り込み"
  )

  # ステップ3: 詳細パターン検出
  st.subheader("ステップ2: 詳細パターン検出")
  _run_normal_scan(passed_tickers, period_months, params)


def _run_normal_scan(
  stock_list: dict,
  period_months: int,
  params: DetectionParams,
):
  """通常のスキャン処理（進捗バー付き）"""
  results = []
  progress = st.progress(0, text="スキャン中...")
  total = len(stock_list)

  for i, (ticker, name) in enumerate(stock_list.items()):
    progress.progress(
      (i + 1) / total,
      text=f"詳細分析中... {ticker} ({i + 1}/{total})",
    )

    df = fetch_stock_data(ticker, period_months)
    if df is None or df.empty:
      continue

    result = detect_cup_with_handle(df, params)
    score = calculate_score(result)

    # スコアが0より大きい場合のみ表示
    if score.total > 0 or result.pattern_found:
      current_price = df["close"].iloc[-1]
      results.append({
        "銘柄コード": ticker,
        "銘柄名": name,
        "スコア": round(score.total, 1),
        "ステータス": result.status,
        "現在価格": f"{current_price:,.2f}",
        "目標価格": f"{result.target_price:,.2f}" if result.target_price else "-",
      })

  progress.empty()

  if not results:
    st.warning("カップウィズハンドルパターンの候補は見つかりませんでした。")
    return

  # スコア順でソート
  results_df = pd.DataFrame(results)
  results_df = results_df.sort_values("スコア", ascending=False).reset_index(drop=True)

  st.success(f"{len(results)} 銘柄の候補が見つかりました")
  st.dataframe(
    results_df,
    hide_index=True,
    use_container_width=True,
  )

  # 個別銘柄の詳細表示
  st.subheader("個別銘柄の詳細")
  selected_ticker = st.selectbox(
    "詳細を表示する銘柄を選択",
    results_df["銘柄コード"].tolist(),
  )

  if selected_ticker:
    df = fetch_stock_data(selected_ticker, period_months)
    if df is not None:
      result = detect_cup_with_handle(df, params)
      score = calculate_score(result)
      stock_name = results_df[results_df["銘柄コード"] == selected_ticker]["銘柄名"].iloc[0]

      col1, col2 = st.columns(2)
      with col1:
        st.metric("ステータス", result.status)
      with col2:
        st.metric("スコア", f"{score.total:.1f} / 100")

      fig = create_chart(df, result, selected_ticker, stock_name)
      st.plotly_chart(fig, use_container_width=True)


# --- サイドバー ---
with st.sidebar:
  st.header("設定")

  mode = st.radio(
    "分析モード",
    ["単一銘柄分析", "一括スクリーニング"],
  )

  market = st.selectbox("市場", ["日本株", "米国株"])

  period_options = {"6ヶ月": 6, "1年": 12, "2年": 24}
  period_label = st.selectbox("分析期間", list(period_options.keys()), index=1)
  period_months = period_options[period_label]

  # 上級者向けパラメータ
  with st.expander("パラメータ調整（上級者向け）"):
    cup_depth_min = st.slider(
      "カップ深さ（最小%）", 5, 30, 15, step=1,
    ) / 100
    cup_depth_max = st.slider(
      "カップ深さ（最大%）", 30, 70, 50, step=1,
    ) / 100
    handle_depth_max = st.slider(
      "ハンドル深さ（最大%）", 5, 30, 15, step=1,
    ) / 100
    cup_period_min = st.slider(
      "カップ期間（最小日数）", 10, 60, 30, step=5,
    )
    cup_period_max = st.slider(
      "カップ期間（最大日数）", 60, 300, 150, step=10,
    )

  params = DetectionParams(
    cup_depth_min=cup_depth_min,
    cup_depth_max=cup_depth_max,
    handle_depth_max=handle_depth_max,
    cup_period_min=cup_period_min,
    cup_period_max=cup_period_max,
  )

# --- メイン画面 ---
if mode == "単一銘柄分析":
  render_single_mode(market, period_months, params)
else:
  render_screening_mode(market, period_months, params)

# --- フッター ---
st.divider()
st.caption(
  "⚠️ 本ツールは投資判断の補助を目的としており、投資の成果を保証するものではありません。"
  "投資に関する最終的な判断は、ご自身の責任において行ってください。"
)
