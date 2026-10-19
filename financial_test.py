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

# =========================================================
# 2. 作業二核心 MPT 引擎（自動讀取理財機器人投資組合.csv）
# =========================================================
@st.cache_data
def load_and_compute_mpt():
    file_path = "理財機器人投資組合.csv"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            encoding = chardet.detect(f.read(100000))["encoding"]
        try:
            df = pd.read_csv(file_path, encoding=encoding, sep="\t")
            if "收盤價(元)" not in df.columns:
                df = pd.read_csv(file_path, encoding=encoding)
        except Exception:
            df = pd.read_csv(file_path)

        df["年月日"] = pd.to_datetime(df["年月日"], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["年月日"])
        df["收盤價(元)"] = pd.to_numeric(df["收盤價(元)"], errors="coerce")
        data = df.pivot(index="年月日", columns="證券代碼", values="收盤價(元)").sort_index().ffill()
        returns = data.pct_change().dropna(how="any")

        assets = list(returns.columns)
        mean_ret = returns.mean().values * TRADING_DAYS
        cov_matrix = returns.cov().values * TRADING_DAYS
        return assets, mean_ret, cov_matrix

ASSETS, MEAN_RETURNS, COV_MATRIX = load_and_compute_mpt()
NUM_ASSETS = len(ASSETS)

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
            line=dict(color='#000000', width=2)
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#64748b"),
                angularaxis=dict(color="#cbd5e1"),
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

    #=====================================================
    # 功能 1：效率前緣圖 (加上保守/穩健/積極風險區間背景與高亮)
    #=====================================================
    st.subheader("📈 1. 效率前緣曲線與最佳化落點 (Efficient Frontier)")
    st.caption("背景區塊已標註三大風險屬性區間，醒目發光區域代表與您投資者輪廓相符之推薦落點範圍。")

    fig_ef = go.Figure()

    # 計算風險區間範圍 (以波動率 Volatility % 為基準進行劃分)
    all_vols = [v * 100 for v in ef_vols] + [mvp_vol * 100, tan_vol * 100, opt_vol * 100]
    min_v = min(all_vols)
    max_v = max(all_vols)
    span = max_v - min_v
    x_start = max(0.0, min_v - span * 0.05)
    x_end = max_v + span * 0.08
    cut1 = x_start + (x_end - x_start) / 3.0
    cut2 = x_start + (x_end - x_start) * 2.0 / 3.0

    risk_zones = [
        {"name": "保守型", "x0": x_start, "x1": cut1, "color": "#10b981", "type_key": "保守型"},
        {"name": "穩健型", "x0": cut1, "x1": cut2, "color": "#06b6d4", "type_key": "穩健型"},
        {"name": "積極型", "x0": cut2, "x1": x_end, "color": "#f43f5e", "type_key": "積極型"}
    ]

    # 加入背景區間遮罩 (當前對應類型套用高對比醒目色彩與實線框)
    for rz in risk_zones:
        is_matched = (rz["type_key"]in investor_type)
        if is_matched:
            # 醒目高亮樣式
            if rz["type_key"] == "保守型":
                bg_color = "rgba(16, 185, 129, 0.25)"
            elif rz["type_key"] == "穩健型":
                bg_color = "rgba(6, 182, 212, 0.25)"
            else:
                bg_color = "rgba(244, 63, 94, 0.25)"

            fig_ef.add_vrect(
                x0=rz["x0"], x1=rz["x1"],
                fillcolor=bg_color,
                layer="below",
                line=dict(color=rz["color"], width=2.5),
                annotation_text=f"<b>★ 您的屬性區間：{rz['name']} ★</b>",
                annotation_position="top left",
                annotation_font=dict(color=rz["color"], size=13, family="Segoe UI")
            )
        else:
            # 非所屬類型淡化背景
            fig_ef.add_vrect(
                x0=rz["x0"], x1=rz["x1"],
                fillcolor="rgba(148, 163, 184, 0.07)",
                layer="below",
                line=dict(color="rgba(148, 163, 184, 0.3)", width=1, dash="dot"),
                annotation_text=f"{rz['name']}風險區",
                annotation_position="top left",
                annotation_font=dict(color="#94a3b8", size=11, family="Segoe UI")
            )

    # 效率前緣曲線
    fig_ef.add_trace(go.Scatter(
        x=[v * 100 for v in ef_vols], y=[r * 100 for r in ef_rets],
        mode='lines', name='效率前緣 (Efficient Frontier)',
        line=dict(color='#0284c7', width=3)
    ))
    # MVP 點
    fig_ef.add_trace(go.Scatter(
        x=[mvp_vol * 100], y=[mvp_r * 100],
        mode='markers+text', name='最小變異組合 (MVP)',
        marker=dict(color='#e11d48', size=12, symbol='diamond'),
        text=['MVP'], textposition='top left'
    ))
    # Tangency 點
    fig_ef.add_trace(go.Scatter(
        x=[tan_vol * 100], y=[tan_r * 100],
        mode='markers+text', name='最高夏普 (Tangency)',
        marker=dict(color='#d97706', size=12, symbol='triangle-up'),
        text=['Tangency'], textposition='bottom right'
    ))
    # 使用者最適落點
    fig_ef.add_trace(go.Scatter(
        x=[opt_vol * 100], y=[opt_r * 100],
        mode='markers+text', name='您的最適配置組合',
        marker=dict(color='#059669', size=16, symbol='star', line=dict(color='#ffffff', width=1.5)),
        text=['★ 最適推薦點'], textposition='top center'
    ))

    fig_ef.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="年化波動率 / 風險 (σ %)",
        yaxis_title="年化預期報酬率 (E(R) %)",
        xaxis=dict(range=[x_start, x_end], showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        height=480,
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center")
    )
    st.plotly_chart(fig_ef, use_container_width=True)
    # =====================================================
    # 功能 2：資金與經驗設定
    # =====================================================
    st.subheader("💰 2. 總投資金額設定")
    total_capital = st.number_input(
        "請填寫預計投入之總金額 (TWD)：",
        min_value=1000, max_value=100000000, value=500000, step=50000
    )

    # =====================================================
    # 功能 3：互動式配置表與甜甜圈圖
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
        st.session_state.user_weights = {ASSETS[i]: int(base_weights[i]) for i in range(NUM_ASSETS)}
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
                    若欲調高特定標的，請先將其他標的之拉桿調低以釋出額度。
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

        # 動態計算每個拉桿的 max_value，嚴格杜絕超過 100%
        for ticker in ASSETS:
            curr_val = st.session_state.user_weights[ticker]
            other_sum = sum(st.session_state.user_weights[t] for t in ASSETS if t != ticker)
            allowed_max = max(curr_val, 100 - other_sum)

            # 當該標的無法再增加時，給出提示文字
            is_locked = (curr_val == allowed_max and current_sum >= 100)
            tag = " 🔒 (已達上限)" if is_locked else ""

            new_val = st.slider(
                f"{ticker}{tag}",
                min_value=0,
                max_value=int(allowed_max),
                value=int(curr_val),
                step=1,
                key=f"slider_{ticker}"
            )
            # 即時同步回 session_state
            st.session_state.user_weights[ticker] = new_val

    # 配置明細表
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
        st.rerun()
        if "user_weights" in st.session_state:
            del st.session_state["user_weights"]
        st.rerun()
