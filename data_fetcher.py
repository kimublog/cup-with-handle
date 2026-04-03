"""株価データ取得モジュール（yfinanceラッパー）"""
from __future__ import annotations

import time
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# レート制限時のリトライ設定
MAX_RETRIES = 3
RETRY_WAIT_SEC = 10


def _fetch_history(ticker: str, start_date, end_date) -> pd.DataFrame:
  """レート制限対応付きでyfinance historyを取得する"""
  for attempt in range(MAX_RETRIES):
    try:
      stock = yf.Ticker(ticker)
      df = stock.history(start=start_date, end=end_date)
      return df
    except Exception as e:
      # レート制限エラーの場合はリトライ
      if "rate" in str(e).lower() or "too many" in str(e).lower():
        if attempt < MAX_RETRIES - 1:
          time.sleep(RETRY_WAIT_SEC * (attempt + 1))
          continue
      raise
  return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker: str, period_months: int = 12) -> pd.DataFrame | None:
  """
  yfinanceで株価データ（日足）を取得する。
  1時間キャッシュ付き。

  Args:
    ticker: 銘柄コード（例: "7203.T", "AAPL"）
    period_months: 取得期間（月数）

  Returns:
    株価DataFrameまたはNone（取得失敗時）
  """
  try:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_months * 30)

    df = _fetch_history(ticker, start_date, end_date)

    if df.empty:
      return None

    # カラム名を正規化
    df.columns = [col.lower() for col in df.columns]

    # 必要なカラムが存在するか確認
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
      if col not in df.columns:
        return None

    return df[required]

  except Exception:
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_name(ticker: str) -> str:
  """銘柄名を取得する"""
  try:
    stock = yf.Ticker(ticker)
    info = stock.info
    # 日本株の場合はlongNameまたはshortNameを使用
    name = info.get("longName") or info.get("shortName") or ticker
    return name
  except Exception:
    return ticker


def pre_filter_stock(ticker: str, period_months: int = 12) -> bool:
  """
  事前フィルタ: カップウィズハンドルの候補になり得るかを軽量に判定する。
  条件: 期間中に高値から15%以上下落し、その後底から30%以上回復している。

  Args:
    ticker: 銘柄コード
    period_months: 分析期間（月数）

  Returns:
    True: 候補の可能性あり / False: 候補外
  """
  import numpy as np

  try:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_months * 30)

    df = _fetch_history(ticker, start_date, end_date)

    if df.empty or len(df) < 30:
      return False

    # カラム名を正規化
    df.columns = [col.lower() for col in df.columns]
    if "close" not in df.columns:
      return False

    close = df["close"].values

    # 最高値とそのインデックスを特定
    high_idx = np.argmax(close)
    high_price = close[high_idx]

    # 最高値以降の最安値を探す
    if high_idx >= len(close) - 10:
      return False

    after_high = close[high_idx:]
    low_offset = np.argmin(after_high)
    low_idx = high_idx + low_offset
    low_price = close[low_idx]

    # 15%以上の下落があるか
    drop_rate = (high_price - low_price) / high_price
    if drop_rate < 0.15:
      return False

    # 底以降に回復傾向があるか（底から30%以上回復）
    if low_idx >= len(close) - 3:
      return False

    current_price = close[-1]
    recovery = (current_price - low_price) / (high_price - low_price)
    if recovery < 0.30:
      return False

    return True

  except Exception:
    return False


def run_pre_filter(
  tickers: dict[str, str],
  period_months: int = 12,
  max_workers: int = 5,
  progress_callback: object = None,
) -> dict[str, str]:
  """
  複数銘柄に対して並列で事前フィルタを実行する。
  レート制限を避けるため、バッチ間にウェイトを挟む。

  Args:
    tickers: {ticker: name} の辞書
    period_months: 分析期間
    max_workers: 並列数（デフォルト5、レート制限対策）
    progress_callback: 進捗コールバック関数 (current, total, ticker) -> None

  Returns:
    フィルタを通過した {ticker: name} の辞書
  """
  from concurrent.futures import ThreadPoolExecutor, as_completed

  passed = {}
  total = len(tickers)
  completed = 0
  ticker_list = list(tickers.items())

  # バッチ処理でレート制限を回避（50銘柄ごとに1秒待機）
  batch_size = 50
  for batch_start in range(0, len(ticker_list), batch_size):
    batch = ticker_list[batch_start:batch_start + batch_size]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
      future_to_ticker = {}
      for ticker, name in batch:
        future = executor.submit(pre_filter_stock, ticker, period_months)
        future_to_ticker[future] = (ticker, name)

      for future in as_completed(future_to_ticker):
        ticker, name = future_to_ticker[future]
        completed += 1

        try:
          if future.result():
            passed[ticker] = name
        except Exception:
          pass

        if progress_callback:
          progress_callback(completed, total, ticker)

    # バッチ間のウェイト（レート制限回避）
    if batch_start + batch_size < len(ticker_list):
      time.sleep(1)

  return passed
