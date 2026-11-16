import os
import json
import chardet
import numpy as np
import pandas as pd
import scipy.optimize as sco
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# 1. 網頁全域風格設定 (FinTech 深色科技質感)
# =========================================================
st.set_page_config(
    page_title="理財機器人：數位投資者輪廓與 MPT 最適資產配置系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 注入客製化深色 CSS，徹底擺脫 Streamlit 預設單調外觀
st.markdown("""
<style>
    /* 深色背景與字體 */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    /* 卡片式容器 */
    .card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .badge {
        background: #e0f2fe;
        color: #0369a1;
        border: 1px solid #7dd3fc;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }
    /* 美化單選按鈕 */
    div[role="radiogroup"] > label {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        padding: 12px 18px;
        border-radius: 12px;
        margin-bottom: 10px;
        transition: all 0.2s ease;
    }
    div[role="radiogroup"] > label:hover {
        border-color: #06b6d4;
        background-color: #f0f9ff;
    }
    /* 按鈕樣式 */
    .stButton>button {
        background: linear-gradient(135deg, #06b6d4 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        box-shadow: 0 4px 14px 0 rgba(6, 182, 212, 0.3);
        transition: all 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px 0 rgba(6, 182, 212, 0.45);
    }
</style>
""", unsafe_allow_html=True)

TRADING_DAYS = 252

def load_and_compute_mpt():
    global DATA_SOURCE_FILE
    import glob

    candidates = sorted(glob.glob("*理財機器人投資組合*.xlsx"))
    if not candidates:
        candidates = sorted(glob.glob("*.xlsx"))
    if not candidates:
        raise FileNotFoundError("找不到資產 Excel 檔案。")

    file_path = candidates[0]
    excel = pd.ExcelFile(file_path, engine="openpyxl")
    price_series = {}

    for sheet_name in excel.sheet_names:
        # 利率不是股票資料工作表，略過
        if sheet_name == "利率":
            continue

        sheet = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
        if sheet.empty:
            continue

        date_col = "年月日" if "年月日" in sheet.columns else sheet.columns[0]

        date_text = (
            sheet[date_col].astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )
        dates = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
        if dates.isna().all():
            dates = pd.to_datetime(sheet[date_col], errors="coerce")

        # 優先使用還原／調整收盤價；沒有的話使用一般收盤價
        def close_column_priority(column):
            name = str(column).lower().replace(" ", "").replace("_", "")
            if "還原收盤" in name or "調整收盤" in name or "adjustedclose" in name or "adjclose" in name:
                return 0
            if "收盤價" in name or name == "收盤" or "close" in name:
                return 1
            return 2

        close_columns = [
            column for column in sheet.columns
            if close_column_priority(column) < 2
        ]
        if not close_columns:
            # 例如工作表不是股票價格表時，略過
            continue

        close_col = sorted(close_columns, key=close_column_priority)[0]
        close = pd.to_numeric(sheet[close_col], errors="coerce")

        stock_prices = pd.DataFrame({
            "date": dates,
            "price": close,
        })
        stock_prices = stock_prices.dropna(subset=["date", "price"])
        stock_prices = stock_prices[stock_prices["price"] > 0]
        stock_prices = stock_prices.drop_duplicates("date", keep="last")
        stock_prices = stock_prices.set_index("date").sort_index()

        if not stock_prices.empty:
            # 使用工作表名稱作為股票名稱
            price_series[sheet_name] = stock_prices["price"]

    if not price_series:
        raise ValueError(
            "找不到股票收盤價欄位。請確認每個股票工作表有「年月日」及「收盤價」欄。"
        )

    # 合併各股票收盤價；欄名為股票工作表名稱
    prices = pd.concat(price_series, axis=1).sort_index()
    prices.columns = list(price_series.keys())

    # 由收盤價計算每日報酬率；只保留所有股票都有資料的日期
    returns_df = (
        prices.pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(axis=0, how="any")
    )

    if returns_df.empty or len(returns_df) < 2:
        raise ValueError("有效的多股票收盤價資料不足，無法計算報酬率與共變異數。")

    if not np.isfinite(returns_df.to_numpy()).all():
        raise ValueError("股票報酬資料含有無效數值，請檢查 Excel 收盤價。")

    DATA_SOURCE_FILE = file_path
    assets = returns_df.columns.tolist()
    mean_returns = returns_df.mean().to_numpy() * TRADING_DAYS
    cov_matrix = returns_df.cov().to_numpy() * TRADING_DAYS

    return assets, mean_returns, cov_matrix, returns_df

DATA_SOURCE_FILE = None
ASSETS, MEAN_RETURNS, COV_MATRIX, RETURNS_DF = load_and_compute_mpt()
NUM_ASSETS = len(ASSETS)


@st.cache_data
def load_market_inputs(return_dates):
    """
    從同資料夾 Excel 的「利率」工作表讀取無風險利率與市場風險溢酬。
    無風險利率欄位以年百分比表示；市場風險溢酬欄位視為每日百分比，
    先與資產報酬日期對齊，再以平均每日值 × 252 轉成年化報酬率。
    """
    import glob

    patterns = ("*理財機器人投資組合*.xlsx", "*.xlsx")
    candidates = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if path not in candidates and os.path.isfile(path):
                candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "找不到 Excel 利率資料。請將含有「利率」工作表的 .xlsx 檔案放在 Python 程式同一資料夾。"
        )

    last_error = None
    for file_path in candidates:
        try:
            rate_df = pd.read_excel(file_path, sheet_name="利率", engine="openpyxl")
            required = {"年月日", "無風險利率", "市場風險溢酬"}
            if not required.issubset(rate_df.columns):
                raise ValueError(f"「利率」工作表缺少欄位：{required - set(rate_df.columns)}")

            date_text = rate_df["年月日"].astype(str).str.replace(r"\\.0$", "", regex=True).str.strip()
            rate_df["年月日"] = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
            rate_df["無風險利率"] = pd.to_numeric(rate_df["無風險利率"], errors="coerce")
            rate_df["市場風險溢酬"] = pd.to_numeric(rate_df["市場風險溢酬"], errors="coerce")
            rate_df = rate_df.dropna(subset=["年月日", "無風險利率", "市場風險溢酬"])
            rate_df = rate_df.drop_duplicates("年月日", keep="last").set_index("年月日").sort_index()

            aligned = rate_df.reindex(pd.DatetimeIndex(return_dates)).dropna(
                subset=["無風險利率", "市場風險溢酬"]
            )
            if len(aligned) < 30:
                raise ValueError("Excel 利率資料與資產報酬資料的有效重疊日期不足 30 筆。")

            # TEJ 欄位數值如 0.795、1.700，代表年化百分比；轉成小數。
            risk_free_rate = float(aligned["無風險利率"].mean() / 100.0)
            # 市場風險溢酬欄位為每日百分比；平均後乘 252 轉成年化小數。
            market_risk_premium = float(aligned["市場風險溢酬"].mean() * TRADING_DAYS / 100.0)

            if not np.isfinite(risk_free_rate) or not np.isfinite(market_risk_premium):
                raise ValueError("無風險利率或市場風險溢酬計算結果不是有效數值。")
            return risk_free_rate, market_risk_premium, file_path, len(aligned), aligned.index.min(), aligned.index.max()
        except Exception as exc:
            last_error = exc

    raise ValueError(f"無法讀取 Excel 的「利率」工作表：{last_error}")


try:
    (RISK_FREE_RATE, MARKET_RISK_PREMIUM, RATE_SOURCE_FILE,
     RATE_OBSERVATIONS, RATE_START_DATE, RATE_END_DATE) = load_market_inputs(
         tuple(RETURNS_DF.index.strftime("%Y-%m-%d"))
     )
except Exception as exc:
    st.error(f"無法載入無風險利率／市場風險溢酬：{exc}")
    st.stop()

st.sidebar.caption(f"資產資料：{os.path.basename(DATA_SOURCE_FILE)}")
st.sidebar.caption(f"利率資料：{os.path.basename(RATE_SOURCE_FILE)}")
st.sidebar.caption(f"年化無風險利率：{RISK_FREE_RATE:.2%}｜年化市場風險溢酬：{MARKET_RISK_PREMIUM:.2%}")


def port_perf(weights, mean_returns, cov_matrix):
    r = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return r, vol
    
# 最小變異數組合 (MVP)
def get_mvp():
    bounds = tuple((0.0, 1.0) for _ in range(NUM_ASSETS))
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    init = np.ones(NUM_ASSETS) / NUM_ASSETS
    res = sco.minimize(lambda w: np.dot(w.T, np.dot(COV_MATRIX, w)), init, method='SLSQP', bounds=bounds, constraints=cons)
    r, vol = port_perf(res.x, MEAN_RETURNS, COV_MATRIX)
    return res.x, r, vol

# 最大夏普組合 (Tangency)
def get_tangency(rf=0.015):
    bounds = tuple((0.0, 1.0) for _ in range(NUM_ASSETS))
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    init = np.ones(NUM_ASSETS) / NUM_ASSETS
    def neg_sharpe(w):
        r, vol = port_perf(w, MEAN_RETURNS, COV_MATRIX)
        return -(r - rf) / (vol if vol != 0 else 1e-6)
    res = sco.minimize(neg_sharpe, init, method='SLSQP', bounds=bounds, constraints=cons)
    r, vol = port_perf(res.x, MEAN_RETURNS, COV_MATRIX)
    return res.x, r, vol

# 計算效率前緣曲線
def get_efficient_frontier(points=35):
    _, min_r, _ = get_mvp()
    max_r = np.max(MEAN_RETURNS) * 0.98
    target_rets = np.linspace(min_r, max_r, points)
    vols, rets = [], []
    bounds = tuple((0.0, 1.0) for _ in range(NUM_ASSETS))
    init = np.ones(NUM_ASSETS) / NUM_ASSETS

    for t_ret in target_rets:
        cons = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w: np.dot(w, MEAN_RETURNS) - t_ret}
        )
        res = sco.minimize(lambda w: np.dot(w.T, np.dot(COV_MATRIX, w)), init, method='SLSQP', bounds=bounds, constraints=cons)
        if res.success:
            vols.append(np.sqrt(res.fun))
            rets.append(t_ret)
    return vols, rets

# 作業四：Black-Litterman 模型函數
# =========================================================
def calculate_market_prior(mean_returns, cov_matrix, market_weights,
                           market_risk_premium, risk_free_rate=RISK_FREE_RATE):
    """使用 Excel 提供的年化市場風險溢酬估計 δ，並計算隱含均衡報酬 Π。"""
    weights = np.asarray(market_weights, dtype=float)
    weights = weights / weights.sum()
    mu = np.asarray(mean_returns, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)

    market_return = float(weights @ mu)
    market_variance = float(weights @ cov @ weights)
    if market_variance <= 1e-12:
        raise ValueError("市場投資組合變異數過小，無法估計風險趨避係數。")

    # δ = 市場風險溢酬 / 市場投資組合變異數；兩者均為年化數值。
    raw_delta = float(market_risk_premium) / market_variance
    # δ 必須為正值；若樣本平均市場風險溢酬非正，採用小幅正值下限並揭露。
    delta = max(float(raw_delta), 0.10)
    equilibrium_returns = delta * cov @ weights
    return equilibrium_returns, delta, market_return, market_variance, raw_delta


def build_absolute_views(assets, selected_assets, view_returns, confidences, cov_matrix, tau=0.05):
    """
    建立絕對觀點：
    每一列 P 對應一項資產，Q 為該資產的年化預期報酬。
    Ω 使用信心比例轉換的不確定性近似值；信心越高，Ω 越小。
    """
    if not selected_assets:
        return np.empty((0, len(assets))), np.array([]), np.empty((0, 0))

    p_rows, q_values, omega_values = [], [], []
    cov = np.asarray(cov_matrix, dtype=float)

    for asset in selected_assets:
        idx = assets.index(asset)
        row = np.zeros(len(assets))
        row[idx] = 1.0
        p_rows.append(row)
        q_values.append(float(view_returns[asset]))

        confidence = float(np.clip(confidences[asset], 0.05, 0.95))
        view_variance = float(row @ cov @ row.T)
        # 教學/介面用的信心轉換近似式，不宣稱為唯一標準校準法。
        omega_i = ((1.0 - confidence) / confidence) * tau * max(view_variance, 1e-10)
        omega_values.append(max(omega_i, 1e-10))

    P = np.vstack(p_rows)
    Q = np.asarray(q_values, dtype=float)
    Omega = np.diag(omega_values)
    return P, Q, Omega


def calculate_black_litterman(pi, cov_matrix, P, Q, Omega, tau=0.05):
    """依 BL 後驗報酬公式計算 μ_BL，使用 pseudo-inverse 增加數值穩定性。"""
    pi = np.asarray(pi, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)

    if P.size == 0:
        return pi.copy()

    tau_cov = tau * cov
    tau_cov_inv = np.linalg.pinv(tau_cov)
    omega_inv = np.linalg.pinv(Omega)

    posterior_precision = tau_cov_inv + P.T @ omega_inv @ P
    posterior_cov = np.linalg.pinv(posterior_precision)
    posterior_mean = posterior_cov @ (
        tau_cov_inv @ pi + P.T @ omega_inv @ Q
    )
    return np.asarray(posterior_mean, dtype=float)

def calculate_portfolio_sharpe(portfolio_return, portfolio_vol, risk_free_rate=RISK_FREE_RATE):
    if portfolio_vol <= 1e-12:
        return 0.0
    return (portfolio_return - risk_free_rate) / portfolio_vol

# 根據問卷分數透過效用函數 (Utility Maximization) 動態求解
def optimize_portfolio_for_user(risk_score_norm):
    # A (風險趨避係數)：分數越低，A 越大（保守）；分數越高，A 越小（積極）
    A = 18.0 - (risk_score_norm / 100.0) * (18.0 - 1.5)
    def utility(w):
        r, _ = port_perf(w, MEAN_RETURNS, COV_MATRIX)
        var = np.dot(w.T, np.dot(COV_MATRIX, w))
        return -(r - 0.5 * A * var)
    bounds = tuple((0.0, 1.0) for _ in range(NUM_ASSETS))
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    init = np.ones(NUM_ASSETS) / NUM_ASSETS
    res = sco.minimize(utility, init, method='SLSQP', bounds=bounds, constraints=cons)
    r, vol = port_perf(res.x, MEAN_RETURNS, COV_MATRIX)
    return res.x, r, vol
def optimize_portfolio_with_returns(mean_returns, risk_score_norm, cov_matrix):
    risk_aversion = 18.0 - (risk_score_norm / 100.0) * (18.0 - 1.5)

    def utility(weights):
        portfolio_return = np.dot(weights, mean_returns)
        variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return -(portfolio_return - 0.5 * risk_aversion * variance)

    bounds = tuple((0.0, 1.0) for _ in range(len(mean_returns)))
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},)
    initial_weights = np.ones(len(mean_returns)) / len(mean_returns)

    result = sco.minimize(
        utility,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"投資組合最佳化失敗：{result.message}")

    weights = result.x
    portfolio_return = float(np.dot(weights, mean_returns))
    portfolio_vol = float(np.sqrt(weights.T @ cov_matrix @ weights))
    return weights, portfolio_return, portfolio_vol
# ---------------------------------------------------------
# [新增] 計算三種代表性基準投資人（保守 15分、穩健 50分、積極 85分）
# ---------------------------------------------------------
@st.cache_data
def get_three_benchmark_profiles():
    benchmarks = {
        "Conservative (保守型)": 15.0,
        "Moderate (穩健型)": 50.0,
        "Aggressive (積極型)": 85.0
    }
    results = {}
    for name, score in benchmarks.items():
        w, r, vol = optimize_portfolio_for_user(score)
        results[name] = {"weights": w, "ret": r, "vol": vol, "score": score}
    return results
# =========================================================
# 3. 完整 20 題問卷題庫
# =========================================================
QUESTIONS = [
    {"id": "Q1", "zh": "您整體的投資風險態度為何？", "en": "What is your overall investment risk attitude?",
     "options": [("A", "我非常重視本金安全。", 1), ("B", "我偏好承擔少量風險，以追求穩定成長。", 2), ("C", "我能接受一定程度的波動，以追求長期成長。", 3), ("D", "我能接受較大的波動，以追求較高的長期報酬。", 4)]},
    {"id": "Q2", "zh": "您主要的投資目標是什麼？", "en": "What is your primary investment objective?",
     "options": [("A", "保護本金，避免資產減少。", 1), ("B", "追求穩定收益。", 2), ("C", "追求長期資產成長。", 3), ("D", "追求較高的資本增值。", 4)]},
    {"id": "Q3", "zh": "您預計何時需要使用這筆投資資金？", "en": "When do you expect to need this investment money?",
     "options": [("A", "一年內。", 1), ("B", "超過 1 年至 3 年。", 2), ("C", "超過 3 年至 5 年。", 3), ("D", "超過 5 年。", 4)]},
    {"id": "Q4", "zh": "您可以接受投資組合最多損失多少？", "en": "What is the maximum loss you could tolerate?",
     "options": [("A", "無法接受本金損失。", 1), ("B", "超過 0% 至 5%。", 2), ("C", "超過 5% 至 20%。", 3), ("D", "超過 20%。", 4)]},
    {"id": "Q5", "zh": "假設您的投資組合一年內下跌 20%，您會怎麼做？", "en": "If your portfolio fell 20% in one year, what would you do?",
     "options": [("A", "立即賣出，避免進一步損失。", 1), ("B", "賣出部分資產，降低風險。", 2), ("C", "先持有並觀察市場變化。", 3), ("D", "重新評估財務狀況與投資理由，再考慮加碼。", 4)]},
    {"id": "Q6", "zh": "您對投資商品與市場的熟悉程度如何？", "en": "How familiar are you with investment products?",
     "options": [("A", "幾乎不了解，也沒有投資經驗。", 1), ("B", "了解一些基本概念，經驗有限。", 2), ("C", "了解常見投資商品及其風險。", 3), ("D", "熟悉多種投資商品及其風險。", 4)]},
    {"id": "Q7", "zh": "當市場突然大幅下跌時，您通常會有什麼感受？", "en": "How do you feel when the market drops suddenly?",
     "options": [("A", "非常焦慮，難以專注其他事情。", 4), ("B", "有些擔心，會頻繁查看市場。", 3), ("C", "會在意，但仍能正常思考。", 2), ("D", "大致平靜，不會過度受到影響。", 1)]},
    {"id": "Q8", "zh": "當投資出現虧損時，您通常會怎麼想？", "en": "How do you think when an investment is losing money?",
     "options": [("A", "不願意實現虧損，想等到回本再賣。", 4), ("B", "即使原本理由改變，仍傾向繼續持有。", 3), ("C", "重新檢視投資理由與相關資訊。", 2), ("D", "依照事先設定原則評估是否調整或退出。", 1)]},
    {"id": "Q9", "zh": "當朋友或社群推薦熱門標的時，您通常會怎麼做？", "en": "What do you do when hot investments are recommended?",
     "options": [("A", "立即跟進，避免錯過機會。", 4), ("B", "容易受到影響，可能跟著買進。", 3), ("C", "先查證資訊，再決定是否投資。", 2), ("D", "依照自己的目標與研究結果決定。", 1)]},
    {"id": "Q10", "zh": "您對自己理解投資資訊與判斷風險的信心如何？", "en": "How confident are you in understanding investments?",
     "options": [("A", "沒有信心，通常需要他人協助。", 1), ("B", "有一些信心，但仍需要協助。", 2), ("C", "有信心，也知道自己有哪些不足。", 3), ("D", "非常有信心，能獨立判斷。", 4)]},
    {"id": "Q11", "zh": "您認為自己成功挑選出打敗市場標的的機率如何？", "en": "How likely are you to beat the market?",
     "options": [("A", "不確定。", 1), ("B", "偶爾可以。", 2), ("C", "相當有機會。", 3), ("D", "大多時候都可以。", 4)]},
    {"id": "Q12", "zh": "您平常的投資決策習慣為何？", "en": "What is your investment decision habit?",
     "options": [("A", "常因短期消息或情緒改變決定。", 1), ("B", "有時會偏離原本計畫。", 2), ("C", "通常會查證資訊並依照計畫行動。", 3), ("D", "有明確原則，並定期檢視決策。", 4)]},
    {"id": "Q13", "zh": "您每月的個人收入大約是多少？", "en": "Approximate monthly personal income?",
     "options": [("A", "目前沒有個人收入。", 1), ("B", "新臺幣 1–40,000 元。", 2), ("C", "新臺幣 40,001–80,000 元。", 3), ("D", "新臺幣 80,001 元以上。", 4)]},
    {"id": "Q14", "zh": "扣除近期必要支出後，目前有多少資金可用於投資？", "en": "Funds available for investment?",
     "options": [("A", "目前沒有可投資資金。", 1), ("B", "新臺幣 1–50,000 元。", 2), ("C", "新臺幣 50,001–300,000 元。", 3), ("D", "新臺幣 300,000 元以上。", 4)]},
    {"id": "Q15", "zh": "您每月必要生活支出與還款大約是多少？", "en": "Monthly essential expenses & debts?",
     "options": [("A", "新臺幣 0–10,000 元。", 4), ("B", "新臺幣 10,001–25,000 元。", 3), ("C", "新臺幣 25,001–50,000 元。", 2), ("D", "新臺幣 50,000 元以上。", 1)]},
    {"id": "Q16", "zh": "您目前的總負債大約是多少？", "en": "What is your approximate total debt?",
     "options": [("A", "目前沒有負債。", 4), ("B", "新臺幣 1–100,000 元。", 3), ("C", "新臺幣 100,001–500,000 元。", 2), ("D", "新臺幣 500,000 元以上。", 1)]},
    {"id": "Q17", "zh": "如果突然需要支付 30,000 元急用支出，您能如何應付？", "en": "How would you handle sudden 30k NTD expense?",
     "options": [("A", "無法應付，需要額外借貸。", 1), ("B", "可以應付，但需延後其他支出。", 2), ("C", "可以現有資金支付，但資金明顯減少。", 3), ("D", "可輕鬆應付，不影響原本規劃。", 4)]},
    {"id": "Q18", "zh": "您目前的可投資資金，多久之內不需要拿來應急？", "en": "How long can funds remain untouched?",
     "options": [("A", "不到一年。", 1), ("B", "超過 1 年至 3 年。", 2), ("C", "超過 3 年至 5 年。", 3), ("D", "超過 5 年。", 4)]},
    {"id": "Q19", "zh": "這筆資金的主要用途是？", "en": "Primary purpose of this investment?",
     "options": [("A", "緊急預備金", 1), ("B", "買房頭期款", 2), ("C", "子女教育基金", 3), ("D", "退休資產", 4), ("E", "其他", 2)]},
    {"id": "Q20", "zh": "如果身邊朋友投資報酬率比你高很多，你的反應是？", "en": "Reaction if friends get much higher returns?",
     "options": [("A", "保持原有策略，不受影響。", 1), ("B", "考慮調整投資方式。", 2), ("C", "感到焦慮，想追趕報酬。", 3), ("D", "覺得自己投資失敗。", 4)]}
]

# =========================================================
# 4. 狀態初始化
# =========================================================
if "current_q" not in st.session_state:
    st.session_state.current_q = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# 標題區
st.title("🤖 理財機器人：數位投資者輪廓 × MPT 資產配置系統")
st.caption("結合 Investor Risk Profiling、行為財務學分析與現代投資組合理論（MPT）動態求解最適權重。")

# =========================================================
# 5. 問卷翻頁互動介面 (Wizard)
# =========================================================
if not st.session_state.submitted:
    total_q = len(QUESTIONS)
    curr_idx = st.session_state.current_q
    q_data = QUESTIONS[curr_idx]

    # 進度指示
    progress = (curr_idx + 1) / total_q
    st.progress(progress)
    st.markdown(f"**進度：第 {curr_idx + 1} 題 / 共 {total_q} 題**")

    st.markdown(f"""
    <div class="card">
        <h3 style="color:#38bdf8; margin-bottom: 4px;">{q_data['id']}. {q_data['zh']}</h3>
        <p style="color:#94a3b8; font-size:14px; margin-bottom: 20px;">{q_data['en']}</p>
    </div>
    """, unsafe_allow_html=True)

    # 選項
    opt_labels = [f"{opt[0]}. {opt[1]}" for opt in q_data["options"]]
    opt_score_map = {f"{opt[0]}. {opt[1]}": opt[2] for opt in q_data["options"]}

    prev_selected_score = st.session_state.answers.get(q_data["id"], None)
    default_idx = None
    if prev_selected_score is not None:
        for idx, opt in enumerate(q_data["options"]):
            if opt[2] == prev_selected_score:
                default_idx = idx

    choice = st.radio(
        "請選擇符合您的答案：",
        options=opt_labels,
        index=default_idx,
        key=f"radio_{q_data['id']}",
        label_visibility="collapsed"
    )

    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if curr_idx > 0:
            if st.button("⬅ 上一題"):
                st.session_state.current_q -= 1
                st.rerun()

    with col3:
        if curr_idx < total_q - 1:
            if st.button("下一題 ➡"):
                if choice:
                    st.session_state.answers[q_data["id"]] = opt_score_map[choice]
                    st.session_state.current_q += 1
                    st.rerun()
                else:
                    st.warning("請先選取答案！")
        else:
            if st.button("🚀 提交分析"):
                if choice:
                    st.session_state.answers[q_data["id"]] = opt_score_map[choice]
                    st.session_state.submitted = True
                    st.rerun()

# =========================================================
# 6. MPT 分析結果儀表板
# =========================================================
else:
    ans = st.session_state.answers

    # 計分模型
    risk_raw = sum([ans.get(f"Q{i}", 2) for i in range(1, 6)])
    risk_norm = round((risk_raw - 5) / 15 * 100, 1)

    if risk_norm <= 33.3:
        investor_type = "保守型投資人 (Conservative)"
    elif risk_norm <= 66.6:
        investor_type = "穩健型投資人 (Moderate)"
    else:
        investor_type = "積極型投資人 (Aggressive)"

    q6_raw = ans.get("Q6", 2)
    fam_norm = round((q6_raw - 1) / 3 * 100, 1)
    fin_raw = sum([ans.get(f"Q{i}", 2) for i in range(13, 19)])
    fin_norm = round((fin_raw - 6) / 18 * 100, 1)
    intersection = min(risk_norm, fin_norm, fam_norm)

    # 執行 MPT 數學最佳化
    opt_weights, opt_r, opt_vol = optimize_portfolio_for_user(risk_norm)
    mvp_w, mvp_r, mvp_vol = get_mvp()
    tan_w, tan_r, tan_vol = get_tangency()
    ef_vols, ef_rets = get_efficient_frontier()

    # =====================================================
    # 作業四：Black-Litterman 個人化投資觀點設定
    # =====================================================
    st.markdown("---")
    st.header("🧭 作業四：Black–Litterman 個人化投資觀點")
    st.caption("保留原有 20 題問卷與 MPT；使用者可調整投資觀點及信心水準，再由 BL 模型更新預期報酬。")

    BL_TAU = 0.05
    st.caption(
        f"利率資料期間：{RATE_START_DATE:%Y-%m-%d} 至 {RATE_END_DATE:%Y-%m-%d}，"
        f"有效對齊 {RATE_OBSERVATIONS} 個交易日。無風險利率取對齊期間平均年化值；"
        "市場風險溢酬取每日欄位平均後乘以 252 年化。"    
    )

    with st.expander("① 市場權重設定（Market Portfolio Weights）", expanded=True):
        st.warning(
            "市場權重仍以六項投資標的等權重作為代理（或由使用者自行輸入），不是台灣整體市場的真實市值權重。"
            "無風險利率與市場風險溢酬則改由 Excel「利率」工作表讀取，並依日期與資產報酬資料對齊。"
        )
        market_weight_mode = st.radio(
            "選擇市場權重來源",
            ["等權重代理（預設）", "自行輸入市場權重"],
            horizontal=True,
            key="bl_market_weight_mode"
        )

        if market_weight_mode == "等權重代理（預設）":
            market_weights = np.ones(NUM_ASSETS, dtype=float) / NUM_ASSETS
            st.write("目前每項資產權重：", "、".join([f"{a}: {w:.2%}" for a, w in zip(ASSETS, market_weights)]))
        else:
            st.caption("請輸入各項資產的市場價值占比。程式會將輸入值正規化為總和 100%；請勿將示意數字當成真實市值。")
            entered_weights = []
            cols = st.columns(3)
            for i, asset in enumerate(ASSETS):
                with cols[i % 3]:
                    entered_weights.append(
                        st.number_input(
                            f"{asset} 權重 (%)",
                            min_value=0.0, max_value=100.0,
                            value=round(100.0 / NUM_ASSETS, 2),
                            step=1.0, key=f"bl_mkt_w_{i}"
                        )
                    )
            weight_sum = float(np.sum(entered_weights))
            if weight_sum <= 0:
                st.warning("輸入權重總和為 0，暫時改用等權重代理。")
                market_weights = np.ones(NUM_ASSETS) / NUM_ASSETS
            else:
                market_weights = np.asarray(entered_weights, dtype=float) / weight_sum
                st.caption(f"輸入合計：{weight_sum:.2f}%；正規化後合計：{market_weights.sum():.2%}")

    try:
        pi, bl_delta, market_port_return, market_port_variance, raw_delta = calculate_market_prior(
            MEAN_RETURNS, COV_MATRIX, market_weights,
            MARKET_RISK_PREMIUM, RISK_FREE_RATE
        )
    except Exception as exc:
        st.error(f"市場均衡報酬計算失敗：{exc}")
        pi = MEAN_RETURNS.copy()
        bl_delta = 2.5
        market_port_return = float(market_weights @ MEAN_RETURNS)
        market_port_variance = float(market_weights @ COV_MATRIX @ market_weights)
        raw_delta = np.nan

    st.markdown("#### ② 建立您的投資觀點（Investor Views）")
    st.caption("選擇想表達觀點的資產，輸入未來一年的年化預期報酬率，並設定對該項觀點的信心。這裡採用絕對觀點，較容易理解及驗證。")
    default_views = ASSETS[:min(2, NUM_ASSETS)]
    selected_view_assets = st.multiselect(
        "選擇要設定觀點的資產",
        options=ASSETS,
        default=default_views,
        key="bl_selected_view_assets"
    )

    user_view_returns = {}
    user_confidences = {}
    if selected_view_assets:
        for asset in selected_view_assets:
            idx = ASSETS.index(asset)
            suggested_return = float(np.clip(pi[idx] * 100, -50.0, 100.0))
            with st.container(border=True):
                st.markdown(f"**{asset}**")
                col_view, col_conf = st.columns(2)
                with col_view:
                    q_value = st.number_input(
                        f"{asset} 預期年化報酬率 (%)",
                        min_value=-100.0, max_value=200.0,
                        value=round(suggested_return, 2),
                        step=0.5, key=f"bl_q_{idx}"
                    )
                    if q_value > pi[idx] * 100 + 0.05:
                        direction = "看好（高於市場均衡報酬）"
                    elif q_value < pi[idx] * 100 - 0.05:
                        direction = "看淡（低於市場均衡報酬）"
                    else:
                        direction = "中性（接近市場均衡報酬）"
                with col_conf:
                    confidence = st.slider(
                        f"{asset} 觀點信心水準 (%)",
                        min_value=5, max_value=95, value=60, step=5,
                        key=f"bl_conf_{idx}"
                    )
                    st.caption("信心越高，模型對此項觀點的不確定性設定越低。")
                user_view_returns[asset] = q_value / 100.0
                user_confidences[asset] = confidence / 100.0
                st.caption(f"系統判讀：{direction}；市場均衡報酬為 {pi[idx] * 100:.2f}%。")
    else:
        st.info("目前未選擇任何投資觀點，BL 後驗報酬將等於市場均衡報酬。")
    P, Q, Omega = build_absolute_views(
        ASSETS, selected_view_assets, user_view_returns, user_confidences,
        COV_MATRIX, tau=BL_TAU
    )
    mu_bl = calculate_black_litterman(pi, COV_MATRIX, P, Q, Omega, tau=BL_TAU)

    # MVO 與 BL 使用相同的風險分數、共變異數矩陣、權重限制及最佳化方法。
    bl_weights, bl_opt_return, bl_opt_vol = optimize_portfolio_with_returns(
        mu_bl, risk_norm, COV_MATRIX
    )

    with st.expander("③ 檢視 BL 模型參數與計算結果", expanded=False):
        st.write(f"市場投資組合年化報酬（代理）：**{market_port_return:.2%}**")
        st.write(f"市場投資組合年化變異數：**{market_port_variance:.6f}**")
        st.write(f"Excel 年化市場風險溢酬：**{MARKET_RISK_PREMIUM:.2%}**")
        st.write(f"Excel 平均年化無風險利率：**{RISK_FREE_RATE:.2%}**")
        st.write(f"市場風險趨避係數 δ：**{bl_delta:.4f}**")
        if np.isfinite(raw_delta) and raw_delta <= 0:
            st.info("樣本市場超額報酬不為正，程式將 δ 設定為 0.10 的正值下限，以避免負風險趨避係數。")
        st.write(f"先驗不確定性係數 τ：**{BL_TAU:.2f}**")
        st.markdown("**市場均衡報酬 Π 與 BL 後驗報酬 μ_BL（年化）**")
        df_bl_returns = pd.DataFrame({
            "資產": ASSETS,
            "市場均衡報酬 Π (%)": pi * 100,
            "BL 後驗報酬 μ_BL (%)": mu_bl * 100
        })
        st.dataframe(
            df_bl_returns.style.format({
                "市場均衡報酬 Π (%)": "{:.2f}",
                "BL 後驗報酬 μ_BL (%)": "{:.2f}"
            }),
            use_container_width=True, hide_index=True
        )
        if len(selected_view_assets) > 0:
            st.markdown("**投資觀點矩陣 P、觀點報酬 Q 與不確定性矩陣 Ω**")
            st.write("P：")
            st.dataframe(pd.DataFrame(P, columns=ASSETS, index=selected_view_assets), use_container_width=True)
            st.write("Q（年化報酬）：")
            st.dataframe(pd.DataFrame({"資產": selected_view_assets, "Q (%)": Q * 100}), use_container_width=True, hide_index=True)
            st.write("Ω（觀點不確定性）：")
            st.dataframe(pd.DataFrame(Omega, index=selected_view_assets, columns=selected_view_assets).style.format("{:.6f}"), use_container_width=True)

    # 成果橫幅
    st.markdown(f"""
    <div class="card" style="border-left: 6px solid #06b6d4;">
        <span class="badge">評估結果確認</span>
        <h2 style="color:dark; margin:8px 0;">{investor_type}</h2>
        <p style="color:#ffffff; font-size:14px; margin:0;">
    </div>
    """, unsafe_allow_html=True)

    # 三大指標卡
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("風險偏好 (Attitude)", f"{risk_norm} / 100")
    m2.metric("財務承受力 (Capacity)", f"{fin_norm} / 100")
    m3.metric("投資經驗 (Familiarity)", f"{fam_norm} / 100")
    m4.metric("適性交集邊界", f"{intersection} %")

    # 雷達圖與診斷
    c_left, c_right = st.columns([1, 1])
    with c_left:
        st.subheader("📊 三維度輪廓雷達圖")
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=[risk_norm, fin_norm, fam_norm],
            theta=['風險偏好', '財務承受力', '投資熟悉度'],
            fill='toself',
            fillcolor='rgba(6, 182, 212, 0.25)',
            line=dict(color='#06b6d4', width=2)
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#64748b"),
                angularaxis=dict(color="black"),
                bgcolor="rgba(0,0,0,0)"
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=30, r=30, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with c_right:
        st.subheader("🧠 行為財務學偏差診斷")
        insights = []
        if ans.get("Q7", 1) >= 3:
            insights.append("⚠️ **急跌恐慌傾向**：市場劇烈震盪時容易焦慮，建議拉長檢視週期，避免非理性殺跌。")
        if ans.get("Q8", 1) >= 3:
            insights.append("⚠️ **處置效應（損失規避）**：痛恨實現虧損而容易套牢，進場時務必先設立客觀出場原則。")
        if ans.get("Q9", 1) >= 3:
            insights.append("⚠️ **從眾效應（FOMO 心理）**：容易受熱門話題吸引，請確認配置與長期理財目標一致。")
        if not insights:
            insights.append("✅ **心理素質成熟**：具備高度紀律性，能冷靜遵循量化模型進行長期資產配置。")
        for text in insights:
            st.info(text)

    st.divider()
    # ==================================================
    # [新增] 作業目標 3：資產報酬率、共變異數矩陣與投資組合風險展示
    # =====================================================
    with st.expander("📌 查看核心金融數據：資產報酬率、風險、共變異數矩陣與投資組合風險公式", expanded=False):
        st.markdown("#### 1. 個別標的年化預期報酬率與波動度")
        # 計算個別資產年化標準差
        asset_vols = [np.sqrt(COV_MATRIX[i, i]) for i in range(NUM_ASSETS)]
        df_asset_stat = pd.DataFrame({
            "證券標的代碼": ASSETS,
            "年化預期報酬率 (E(R))": [f"{r * 100:.2f} %" for r in MEAN_RETURNS],
            "年化波動度 (標準差 σ)": [f"{v * 100:.2f} %" for v in asset_vols],
            "個別夏普值 (Rf=1.5%)": [f"{(r - 0.015) / v:.2f}" for r, v in zip(MEAN_RETURNS, asset_vols)]
        })
        st.dataframe(df_asset_stat, use_container_width=True, hide_index=True)

        col_cov, col_corr = st.columns(2)
        with col_cov:
            st.markdown("#### 2. 年化共變異數矩陣 (Covariance Matrix, Σ)")
            st.caption("反映不同資產間價格變動的連動程度（單位：年化變異數）。")
            df_cov = pd.DataFrame(COV_MATRIX, index=ASSETS, columns=ASSETS)
            # 格式化呈現小數點後 5 位
            st.dataframe(df_cov.style.format("{:.5f}"), use_container_width=True)

        with col_corr:
            st.markdown("#### 3. 資產相關係數熱力圖 (Correlation Matrix)")
            st.caption("相關係數介於 -1 到 +1 之間，數值越低越能發揮資產配置的分散風險效果。")
            std_diag = np.sqrt(np.diag(COV_MATRIX))
            corr_matrix = COV_MATRIX / np.outer(std_diag, std_diag)
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=ASSETS,
                y=ASSETS,
                colorscale="Blues",
                zmin=-1, zmax=1,
                text=np.around(corr_matrix, 2),
                texttemplate="%{text}",
                showscale=True
            ))
            fig_corr.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("#### 4. 投資組合風險與報酬計算數學公式驗證")
        st.latex(r"E(R_p) = \mathbf{w}^T \boldsymbol{\mu} = \sum_{i=1}^N w_i E(R_i)")
        st.latex(r"\sigma_p = \sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}} = \sqrt{\sum_{i=1}^N \sum_{j=1}^N w_i w_j \sigma_{ij}}")
        st.info(f"""
        **當前最適投資組合數值實證結果：**
        * 投資組合預期年化報酬率 $E(R_p)$：**{opt_r * 100:.2f} %**
        * 投資組合年化波動風險 $\sigma_p$：**{opt_vol * 100:.2f} %**
        * 夏普比率 (Sharpe Ratio, 無風險利率 1.5%)：**{(opt_r - 0.015) / opt_vol:.2f}**
        """)
    # =====================================================
    # 功能 1：效率前緣圖與三大組合概念解說
    # =====================================================
    # 動態計算「與問卷邊界完全一致」的波動率切點（徹底消除落點與屬性不一致問題）
    _, _, vol_c_edge = optimize_portfolio_for_user(0.0)      # 最極端保守點 (分數 0)
    _, _, vol_cm_cut = optimize_portfolio_for_user(33.33)    # 保守 / 穩健 分界線 (分數 33.3)
    _, _, vol_ma_cut = optimize_portfolio_for_user(66.66)    # 穩健 / 積極 分界線 (分數 66.6)
    _, _, vol_a_edge = optimize_portfolio_for_user(100.0)    # 最極端積極點 (分數 100)

    # 圖表整體 X 軸範疇
    all_vols = [v * 100 for v in ef_vols] + [mvp_vol * 100, tan_vol * 100, opt_vol * 100]
    x_chart_min = min(all_vols) * 0.95
    x_chart_max = max(all_vols) * 1.05

    # 定義精確的三大風險區間（百分比 %）
    risk_zones = [
        {"name": "保守型", "x0": x_chart_min, "x1": vol_cm_cut * 100, "color": "#10b981", "type_key": "保守型"},
        {"name": "穩健型", "x0": vol_cm_cut * 100, "x1": vol_ma_cut * 100, "color": "#06b6d4", "type_key": "穩健型"},
        {"name": "積極型", "x0": vol_ma_cut * 100, "x1": x_chart_max, "color": "#f43f5e", "type_key": "積極型"}
    ]

    # =====================================================
    # [新] Figure 1. 效率前緣與三種投資人落點標註
    # =====================================================
    benchmarks = get_three_benchmark_profiles()

    fig_ef = go.Figure()

    # 1. 背景分區遮罩（保留你原本設計的高質感分區）
    for rz in risk_zones:
        is_matched = (rz["type_key"] in investor_type)
        bg_color = "rgba(6, 182, 212, 0.18)" if is_matched else "rgba(148, 163, 184, 0.05)"
        fig_ef.add_vrect(
            x0=rz["x0"], x1=rz["x1"],
            fillcolor=bg_color, layer="below",
            line=dict(color=rz["color"] if is_matched else "rgba(148, 163, 184, 0.25)", width=2 if is_matched else 1),
            annotation_text=f"{rz['name']}", annotation_position="top left",
            annotation_font=dict(color=rz["color"] if is_matched else "#94a3b8", size=11)
        )

    # 2. 效率前緣曲線
    fig_ef.add_trace(go.Scatter(
        x=[v * 100 for v in ef_vols], y=[r * 100 for r in ef_rets],
        mode='lines', name='效率前緣 (Efficient Frontier)',
        line=dict(color='#0284c7', width=3.5)
    ))

    # 3. 作業核心要求：標記三大標準投資人類型落點 (Figure 1)
    pts_config = {
        "Conservative (保守型)": {"color": "#10b981", "symbol": "circle", "pos": "top left"},
        "Moderate (穩健型)": {"color": "#06b6d4", "symbol": "square", "pos": "top center"},
        "Aggressive (積極型)": {"color": "#f43f5e", "symbol": "diamond", "pos": "bottom right"}
    }
    for p_name, cfg in pts_config.items():
        data = benchmarks[p_name]
        fig_ef.add_trace(go.Scatter(
            x=[data["vol"] * 100], y=[data["ret"] * 100],
            mode='markers+text', name=p_name,
            marker=dict(color=cfg["color"], size=13, symbol=cfg["symbol"], line=dict(color="#ffffff", width=1.5)),
            text=[p_name.split()[0]], textposition=cfg["pos"],
            textfont=dict(color=cfg["color"], size=12, family="Segoe UI")
        ))

    # 4. MVP 與 Tangency 輔助點
    fig_ef.add_trace(go.Scatter(
        x=[mvp_vol * 100], y=[mvp_r * 100],
        mode='markers', name='MVP (最小變異)',
        marker=dict(color='#64748b', size=9, symbol='x')
    ))

    # 5. 當前填答者專屬落點
    fig_ef.add_trace(go.Scatter(
        x=[opt_vol * 100], y=[opt_r * 100],
        mode='markers+text', name='★ 您的評測落點',
        marker=dict(color='#8b5cf6', size=16, symbol='star', line=dict(color='#ffffff', width=2)),
        text=['★ 您的落點'], textposition='top right',
        textfont=dict(color='#8b5cf6', size=13)
    ))

    fig_ef.update_layout(
        title="<b>Figure 1. Efficient Frontier and Investor-Specific Optimal Portfolios</b>",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Portfolio Risk / 年化波動率 (σ %)",
        yaxis_title="Expected Return / 年化預期報酬率 (E(R) %)",
        xaxis=dict(range=[x_chart_min, x_chart_max], showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        height=480,
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center")
    )
    st.plotly_chart(fig_ef, use_container_width=True)
    st.caption("▲ 圖表已標示作業規定之 Conservative、Moderate、Aggressive 三大標準落點，以及您專屬的最適投資組合點。")
    st.subheader("📈 1. 效率前緣曲線與三大經典組合解析")
    # 概念說明卡片
    st.markdown("""
    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; margin-bottom: 20px;">
        <h4 style="margin-top:0; color:#0f172a;">💡 三大經典投資組合是什麼？</h4>
        <ul style="color:#334155; line-height: 1.8; font-size: 14px; margin-bottom: 0;">
            <li><b>♦ 最小變異組合 (MVP, Minimum Variance Portfolio)：</b>全市場<b>波動度最小、風險最低</b>的組合（位於效率前緣最左端）。追求資產抗跌與本金安全，適合無法忍受帳面起伏的保守投資人。</li>
            <li><b>▲ 最高夏普組合 (Tangency Portfolio)：</b>每承擔 1 單位波動風險所換得超額報酬最高的組合（即<b>投資 CP 值最高點</b>）。是現代投資組合理論中資本配置線的切點。</li>
            <li><b>★ 您的最適配置組合 (Optimal Portfolio)：</b>由系統根據您的<b>風險偏好、年齡與資金耐受力</b>，透過二次效用最佳化模型量身算出的落點，追求在您最舒適的心態下獲得最高預期收益。</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 比較差異簡表
    comp_data = {
        "指標 / 組合": ["最小變異組合 (MVP)", "最高夏普組合 (Tangency)", "您的專屬最適組合 (Optimal)"],
        "核心哲學": ["本金安全第一，波動最小", "報酬風險性價比最高 (CP值最高)", "量身定制，心理拿得最安穩"],
        "波動風險 (σ)": [f"{mvp_vol*100:.1f} % (最低)", f"{tan_vol*100:.1f} % (中高)", f"{opt_vol*100:.1f} % (最適化)"],
        "年化預期報酬": [f"{mvp_r*100:.1f} %", f"{tan_r*100:.1f} %", f"{opt_r*100:.1f} %"],
        "適合族群": ["極度保守、即期退休族", "追求市場長期效率之穩健族", f"當前匹配：{investor_type.split()[0]}"]
    }
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
    # =====================================================
    # [新增] Figure 2. 資產配置橫向堆疊比較圖 (作業必備)
    # =====================================================
    st.markdown("---")
    st.subheader("📊 Figure 2. Optimal Asset Allocation by Investor Profile")
    st.caption("由 Python 二次規劃求解之三種風險屬性（Conservative、Moderate、Aggressive）及當前使用者的最適權重分佈對照圖。")

    palette = ['#0284c7', '#0d9488', '#16a34a', '#ca8a04', '#ea580c', '#e11d48', '#9333ea', '#64748b']
    y_categories = ["Conservative", "Moderate", "Aggressive", "★ Your Profile"]
    fig_alloc = go.Figure()

    for idx, ticker in enumerate(ASSETS):
        x_percentages = [
            round(benchmarks["Conservative (保守型)"]["weights"][idx] * 100, 1),
            round(benchmarks["Moderate (穩健型)"]["weights"][idx] * 100, 1),
            round(benchmarks["Aggressive (積極型)"]["weights"][idx] * 100, 1),
            round(opt_weights[idx] * 100, 1)
        ]
        fig_alloc.add_trace(go.Bar(
            y=y_categories,
            x=x_percentages,
            name=str(ticker),
            orientation='h',
            marker=dict(color=palette[idx % len(palette)]),
            text=[f"{v:.1f}%" if v >= 5 else "" for v in x_percentages],
            textposition='inside',
            insidetextanchor='middle'
        ))

    fig_alloc.update_layout(
        barmode='stack',
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="資產配置權重百分比 (%)", range=[0, 100], showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(autorange="reversed"),
        height=320,
        margin=dict(l=20, r=20, t=30, b=30),
        legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center")
    )
    st.plotly_chart(fig_alloc, use_container_width=True)
    st.markdown("---")

    # =====================================================
    # 作業四：MVO vs Black-Litterman 比較
    # =====================================================
    st.header("📊 MVO 與 Black–Litterman 模型比較")
    st.caption("兩種模型採用相同歷史共變異數矩陣、相同問卷風險趨避係數、相同不放空限制；主要差異為預期報酬來源。")

    mvo_sharpe = calculate_portfolio_sharpe(opt_r, opt_vol, RISK_FREE_RATE)
    bl_sharpe = calculate_portfolio_sharpe(bl_opt_return, bl_opt_vol, RISK_FREE_RATE)

    metric_cols = st.columns(3)
    metric_cols[0].metric("MVO 年化預期報酬", f"{opt_r:.2%}")
    metric_cols[1].metric("BL 年化預期報酬", f"{bl_opt_return:.2%}", f"{(bl_opt_return-opt_r)*100:+.2f} 個百分點")
    metric_cols[2].metric("BL 年化波動率", f"{bl_opt_vol:.2%}", f"{(bl_opt_vol-opt_vol)*100:+.2f} 個百分點")
    comparison_table = pd.DataFrame({
        "績效指標": ["年化預期報酬", "年化波動率", "Sharpe Ratio"],
        "傳統 MVO": [f"{opt_r:.2%}", f"{opt_vol:.2%}", f"{mvo_sharpe:.3f}"],
        "Black–Litterman": [f"{bl_opt_return:.2%}", f"{bl_opt_vol:.2%}", f"{bl_sharpe:.3f}"]
    })
    st.dataframe(comparison_table, use_container_width=True, hide_index=True)

    col_compare1, col_compare2 = st.columns(2)
    with col_compare1:
        fig_weights_compare = go.Figure()
        fig_weights_compare.add_trace(go.Bar(
            x=ASSETS, y=opt_weights * 100, name="傳統 MVO",
            marker_color="#0284c7"
        ))
        fig_weights_compare.add_trace(go.Bar(
            x=ASSETS, y=bl_weights * 100, name="Black–Litterman",
            marker_color="#10b981"
        ))
        fig_weights_compare.update_layout(
            title="MVO vs BL 最適資產權重",
            barmode="group", template="plotly_white",
            yaxis_title="配置權重 (%)", xaxis_title="資產",
            height=390, legend=dict(orientation="h", y=-0.25)
        )
        st.plotly_chart(fig_weights_compare, use_container_width=True)

    with col_compare2:
        fig_returns_compare = go.Figure()
        fig_returns_compare.add_trace(go.Bar(
            x=ASSETS, y=pi * 100, name="市場均衡報酬 Π",
            marker_color="#64748b"
        ))
        fig_returns_compare.add_trace(go.Bar(
            x=ASSETS, y=mu_bl * 100, name="BL 後驗報酬",
            marker_color="#8b5cf6"
        ))
        fig_returns_compare.update_layout(
            title="市場均衡報酬 vs BL 後驗報酬",
            barmode="group", template="plotly_white",
            yaxis_title="年化預期報酬 (%)", xaxis_title="資產",
            height=390, legend=dict(orientation="h", y=-0.25)
        )
        st.plotly_chart(fig_returns_compare, use_container_width=True)

    st.markdown("#### 信心水準敏感度分析")
    st.caption("固定目前投資觀點 Q 與市場權重，將所有觀點的信心水準統一調整為 30%、60%、90%，觀察 BL 權重如何變化。")
    sensitivity_rows = []
    sensitivity_confidences = [0.30, 0.60, 0.90]
    if selected_view_assets:
        for conf_level in sensitivity_confidences:
            sensitivity_map = {asset: conf_level for asset in selected_view_assets}
            P_s, Q_s, O_s = build_absolute_views(
                ASSETS, selected_view_assets, user_view_returns,
                sensitivity_map, COV_MATRIX, tau=BL_TAU
            )
            mu_s = calculate_black_litterman(pi, COV_MATRIX, P_s, Q_s, O_s, tau=BL_TAU)
            w_s, r_s, v_s = optimize_portfolio_with_returns(mu_s, risk_norm, COV_MATRIX)
            row = {
                "觀點信心水準": f"{conf_level:.0%}",
                "組合年化預期報酬": f"{r_s:.2%}",
                "組合年化波動率": f"{v_s:.2%}"
            }
            for i, asset in enumerate(ASSETS):
                row[f"{asset} 權重"] = f"{w_s[i]:.1%}"
            sensitivity_rows.append(row)
        st.dataframe(pd.DataFrame(sensitivity_rows), use_container_width=True, hide_index=True)
    else:
        st.info("請至少選擇一項投資觀點，才能進行信心水準敏感度分析。")

    st.info(
        "解讀提醒：BL 的後驗報酬是模型估計值，不代表未來實際報酬保證。"
        "本版本使用絕對觀點及簡化的信心—Ω 轉換方式；若要進行嚴謹實證研究，"
        "應另外校準觀點不確定性，並使用樣本外回測比較績效。"
    )

    st.markdown("---")
    
    # =====================================================
    # 功能 2：資金與經驗設定
    # =====================================================
    st.subheader("💰 2. 總投資金額設定")
    total_capital = st.number_input(
        "請填寫預計投入之總金額 (TWD)：",
        min_value=1000, max_value=100000000, value=500000, step=50000
    )

   # =====================================================
    # 功能 3：互動式配置表與甜甜圈圖（嚴格鎖定 100% 機制 - 防呆修復版）
    # =====================================================
    st.subheader("🥧 3. 最適資產配置權重表與圓餅圖")
    st.caption("權重總和嚴格限定為 100%。若拉動拉桿發現無法增加，代表目前總權重已滿，需先調降其他標的釋出額度。")

    # 初始化配置權重（首次載入時以最適解進行整數配平，確保初始總和必定為 100）
    if "user_weights" not in st.session_state or st.session_state.get("cached_risk_norm") != risk_norm:
        raw_pct = opt_weights * 100.0
        base_weights = np.floor(raw_pct).astype(int)
        remainder = int(100 - np.sum(base_weights))
        fractions = raw_pct - base_weights
        sorted_indices = np.argsort(-fractions)
        for i in range(remainder):
            base_weights[sorted_indices[i]] += 1
        
        st.session_state.user_weights = {}
        for i, ticker in enumerate(ASSETS):
            w = int(base_weights[i])
            st.session_state.user_weights[ticker] = w
            st.session_state[f"slider_{ticker}"] = w
        st.session_state.cached_risk_norm = risk_norm

    col_sliders, col_chart = st.columns([1, 1])

    with col_sliders:
        st.write("##### 各標的配置比例 (可微調)：")
        current_sum = sum(st.session_state.user_weights.values())

        # 即時狀態與鎖定警示
        if current_sum >= 100:
            st.markdown("""
            <div style="background-color: #fef2f2; border: 1.5px solid #ef4444; border-radius: 10px; padding: 10px 14px; margin-bottom: 14px;">
                <span style="color: #b91c1c; font-weight: 700; font-size: 13.5px;">
                    🔒 目前總權重已達 100%（上限已鎖定）
                </span><br>
                <span style="color: #7f1d1d; font-size: 12.5px;">
                    若欲調高特定標的，請先將其他標的之拉桿調低以釋出可用額度。
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            remaining = 100 - current_sum
            st.markdown(f"""
            <div style="background-color: #f0fdf4; border: 1.5px solid #22c55e; border-radius: 10px; padding: 10px 14px; margin-bottom: 14px;">
                <span style="color: #15803d; font-weight: 700; font-size: 13.5px;">
                    ⚡ 目前總權重：{current_sum}%（尚有 {remaining}% 可供分配）
                </span>
            </div>
            """, unsafe_allow_html=True)

        # 動態計算每個拉桿的 max_value，避免 min_value == max_value 錯誤
        for ticker in ASSETS:
            slider_key = f"slider_{ticker}"
            curr_val = st.session_state.user_weights.get(ticker, 0)
            other_sum = sum(st.session_state.user_weights[t] for t in ASSETS if t != ticker)
            allowed_max = max(curr_val, 100 - other_sum)

            # 情況 A：額度為 0 且無法調高（使用 disabled 鎖定，避免 min == max 報錯）
            if allowed_max <= 0:
                st.slider(
                    f"{ticker} 🔒 (目前為 0%，總額已滿無可用額度)",
                    min_value=0,
                    max_value=100,
                    value=0,
                    disabled=True,
                    key=f"{slider_key}_disabled"
                )
                st.session_state.user_weights[ticker] = 0
            
            # 情況 B：正常可調節（allowed_max >= 1）
            else:
                is_capped = (curr_val == allowed_max and current_sum >= 100)
                tag = " 🔒 (已達上限)" if is_capped else ""

                # 防止 session_state 內部值暫時大於 allowed_max
                if slider_key in st.session_state and st.session_state[slider_key] > allowed_max:
                    st.session_state[slider_key] = int(allowed_max)

                new_val = st.slider(
                    f"{ticker}{tag}",
                    min_value=0,
                    max_value=int(allowed_max),
                    value=int(curr_val),
                    step=1,
                    key=slider_key
                )
                st.session_state.user_weights[ticker] = new_val

    # 彙整即時明細表
    portfolio_rows = []
    for ticker in ASSETS:
        w = st.session_state.user_weights[ticker]
        amount = int(round(total_capital * (w / 100.0)))
        portfolio_rows.append({"證券標的": ticker, "配置權重 (%)": f"{w} %", "配置金額 (TWD)": f"{amount:,} 元", "數值權重": w})
    df_port = pd.DataFrame(portfolio_rows)

    with col_chart:
        fig_donut = go.Figure(data=[go.Pie(
            labels=df_port["證券標的"],
            values=df_port["數值權重"],
            hole=0.6,
            textinfo="label+percent",
            marker=dict(colors=['#06b6d4', '#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ec4899'])
        )])
        fig_donut.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=340,
            annotations=[dict(text=f"總額<br><b>{int(total_capital/10000)}萬</b>", x=0.5, y=0.5, font_size=15, showarrow=False)]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.write("##### 資產配置明細清單")
    st.dataframe(df_port[["證券標的", "配置權重 (%)", "配置金額 (TWD)"]], use_container_width=True, hide_index=True)

    # 重新測驗
    if st.button("🔄 重新進行風險問卷測驗"):
        st.session_state.submitted = False
        st.session_state.current_q = 0
        st.session_state.answers = {}
        if "user_weights" in st.session_state:
            del st.session_state["user_weights"]
        for ticker in ASSETS:
            if f"slider_{ticker}" in st.session_state:
                del st.session_state[f"slider_{ticker}"]
        st.rerun()
