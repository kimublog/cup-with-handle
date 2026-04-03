"""銘柄リスト定義モジュール"""
from __future__ import annotations

# 日経225 主要銘柄（代表的な50銘柄）
NIKKEI_225_MAJOR = {
  "7203.T": "トヨタ自動車",
  "6758.T": "ソニーグループ",
  "6861.T": "キーエンス",
  "9984.T": "ソフトバンクグループ",
  "6902.T": "デンソー",
  "8306.T": "三菱UFJフィナンシャル・グループ",
  "8316.T": "三井住友フィナンシャルグループ",
  "8411.T": "みずほフィナンシャルグループ",
  "9433.T": "KDDI",
  "9432.T": "日本電信電話",
  "6501.T": "日立製作所",
  "6098.T": "リクルートホールディングス",
  "7741.T": "HOYA",
  "4063.T": "信越化学工業",
  "6954.T": "ファナック",
  "7974.T": "任天堂",
  "4519.T": "中外製薬",
  "4568.T": "第一三共",
  "6367.T": "ダイキン工業",
  "6273.T": "SMC",
  "8035.T": "東京エレクトロン",
  "6857.T": "アドバンテスト",
  "7267.T": "ホンダ",
  "7751.T": "キヤノン",
  "6762.T": "TDK",
  "4661.T": "オリエンタルランド",
  "9983.T": "ファーストリテイリング",
  "6981.T": "村田製作所",
  "3382.T": "セブン&アイ・ホールディングス",
  "2914.T": "日本たばこ産業",
  "4502.T": "武田薬品工業",
  "4503.T": "アステラス製薬",
  "6594.T": "日本電産",
  "7832.T": "バンダイナムコホールディングス",
  "6645.T": "オムロン",
  "6526.T": "ソシオネクスト",
  "6920.T": "レーザーテック",
  "4911.T": "資生堂",
  "7011.T": "三菱重工業",
  "6301.T": "小松製作所",
  "5108.T": "ブリヂストン",
  "2802.T": "味の素",
  "8001.T": "伊藤忠商事",
  "8058.T": "三菱商事",
  "8031.T": "三井物産",
  "9101.T": "日本郵船",
  "2413.T": "エムスリー",
  "6988.T": "日東電工",
  "4543.T": "テルモ",
  "6723.T": "ルネサスエレクトロニクス",
}

# S&P500 主要銘柄（代表的な50銘柄）
SP500_MAJOR = {
  "AAPL": "Apple",
  "MSFT": "Microsoft",
  "AMZN": "Amazon",
  "NVDA": "NVIDIA",
  "GOOGL": "Alphabet (Class A)",
  "META": "Meta Platforms",
  "TSLA": "Tesla",
  "BRK-B": "Berkshire Hathaway",
  "UNH": "UnitedHealth Group",
  "JNJ": "Johnson & Johnson",
  "V": "Visa",
  "XOM": "Exxon Mobil",
  "JPM": "JPMorgan Chase",
  "MA": "Mastercard",
  "PG": "Procter & Gamble",
  "HD": "Home Depot",
  "AVGO": "Broadcom",
  "CVX": "Chevron",
  "MRK": "Merck",
  "LLY": "Eli Lilly",
  "ABBV": "AbbVie",
  "PEP": "PepsiCo",
  "KO": "Coca-Cola",
  "COST": "Costco",
  "ADBE": "Adobe",
  "WMT": "Walmart",
  "CRM": "Salesforce",
  "MCD": "McDonald's",
  "CSCO": "Cisco",
  "ACN": "Accenture",
  "TMO": "Thermo Fisher",
  "ABT": "Abbott Laboratories",
  "DHR": "Danaher",
  "NEE": "NextEra Energy",
  "LIN": "Linde",
  "TXN": "Texas Instruments",
  "ORCL": "Oracle",
  "AMD": "AMD",
  "NFLX": "Netflix",
  "INTC": "Intel",
  "PM": "Philip Morris",
  "UPS": "UPS",
  "RTX": "RTX Corporation",
  "HON": "Honeywell",
  "LOW": "Lowe's",
  "QCOM": "Qualcomm",
  "CAT": "Caterpillar",
  "AMAT": "Applied Materials",
  "GS": "Goldman Sachs",
  "ISRG": "Intuitive Surgical",
}


def get_stock_list(market: str) -> dict[str, str]:
  """市場に応じた銘柄リストを返す"""
  if market == "日本株":
    return NIKKEI_225_MAJOR
  elif market == "米国株":
    return SP500_MAJOR
  return {}


def fetch_all_jpx_stocks() -> dict[str, str]:
  """
  JPX（日本取引所グループ）の上場銘柄一覧を取得する。
  ETF/REIT等を除く内国株式のみを返す。

  Returns:
    {ticker: 銘柄名} の辞書（例: {"1301.T": "極洋", ...}）
  """
  import urllib.request
  import pandas as pd
  import tempfile
  import os

  url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"

  try:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    response = urllib.request.urlopen(req, timeout=30)

    # 一時ファイルに保存して読み込み
    tmp_path = os.path.join(tempfile.gettempdir(), "jpx_stock_list.xls")
    with open(tmp_path, "wb") as f:
      f.write(response.read())

    df = pd.read_excel(tmp_path)

    # 内国株式のみ抽出（ETF, REIT, 外国株式等を除外）
    stocks = df[df["市場・商品区分"].str.contains("内国株式", na=False)]

    result = {}
    for _, row in stocks.iterrows():
      code = str(row["コード"])
      name = str(row["銘柄名"]).strip()
      ticker = f"{code}.T"
      result[ticker] = name

    return result

  except Exception as e:
    print(f"JPX銘柄一覧の取得に失敗しました: {e}")
    return {}


def format_ticker(ticker: str, market: str) -> str:
  """銘柄コードをyfinance用にフォーマットする"""
  ticker = ticker.strip().upper()
  if market == "日本株" and not ticker.endswith(".T"):
    # 数字のみの場合は.Tを付与
    if ticker.isdigit():
      ticker = f"{ticker}.T"
    # 英数混合コード（例: 130A）にも.Tを付与
    elif not ticker.endswith(".T"):
      ticker = f"{ticker}.T"
  return ticker
