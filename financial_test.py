import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# ==================================================
# 1. 網頁設定 + 標題
# ==================================================
st.set_page_config(
    page_title="Digital Investor Profiling System",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.main-title-zh {
    font-size: 40px;
    font-weight: 800;
    line-height: 2;
}
.main-title-en {
    font-size: 20px;
    color: gray;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

# 只顯示一次標題
st.markdown('<div class="main-title-zh">📊 數位投資者輪廓分析系統</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title-en">Digital Investor Profiling System</div>', unsafe_allow_html=True)

# 說明文字
st.write("透過問卷了解您的風險偏好、行為特徵與財務承受能力。")
st.info("本系統為課程專題原型，結果僅供學習與自我了解，不構成個人化投資建議。")
# =========================================================
# 3. 問卷題目資料
# 每個選項格式：(選項代碼, 中文, 英文, 分數)
# =========================================================

QUESTIONS = [

    # -----------------------------------------------------
    # Part 1: Traditional Risk Profiling
    # -----------------------------------------------------

    {
        "id": "Q1",
        "section": "risk",
        "zh": "您整體的投資風險態度為何？",
        "en": "What is your overall investment risk attitude?",
        "options": [
            ("A", "我非常重視本金安全。",
             "I strongly prioritize protecting my principal.", 1),
            ("B", "我偏好承擔少量風險，以追求穩定成長。",
             "I prefer limited risk for stable growth.", 2),
            ("C", "我能接受一定程度的波動，以追求長期成長。",
             "I accept some volatility for long-term growth.", 3),
            ("D", "我能接受較大的波動，以追求較高的長期報酬。",
             "I accept significant volatility for higher long-term returns.", 4)
        ]
    },

    {
        "id": "Q2",
        "section": "risk",
        "zh": "您主要的投資目標是什麼？",
        "en": "What is your primary investment objective?",
        "options": [
            ("A", "保護本金，避免資產減少。",
             "Protect my principal and avoid losses.", 1),
            ("B", "追求穩定收益。",
             "Seek stable income.", 2),
            ("C", "追求長期資產成長。",
             "Seek long-term capital growth.", 3),
            ("D", "追求較高的資本增值。",
             "Seek higher capital appreciation.", 4)
        ]
    },

    {
        "id": "Q3",
        "section": "risk",
        "zh": "您預計何時需要使用這筆投資資金？",
        "en": "When do you expect to need this investment money?",
        "options": [
            ("A", "一年內。",
             "Within 1 year.", 1),
            ("B", "超過 1 年至 3 年。",
             "More than 1 year, up to 3 years.", 2),
            ("C", "超過 3 年至 5 年。",
             "More than 3 years, up to 5 years.", 3),
            ("D", "超過 5 年。",
             "More than 5 years.", 4)
        ]
    },

    {
        "id": "Q4",
        "section": "risk",
        "zh": "您可以接受投資組合最多損失多少？",
        "en": "What is the maximum loss you could tolerate in your portfolio?",
        "options": [
            ("A", "無法接受本金損失。",
             "I cannot tolerate losing principal.", 1),
            ("B", "超過 0% 至 5%。",
             "More than 0% up to 5%.", 2),
            ("C", "超過 5% 至 20%。",
             "More than 5% up to 20%.", 3),
            ("D", "超過 20%。",
             "More than 20%.", 4)
        ]
    },

    {
        "id": "Q5",
        "section": "risk",
        "zh": "假設您的投資組合一年內下跌 20%，您會怎麼做？",
        "en": "If your portfolio fell by 20% in one year, what would you do?",
        "options": [
            ("A", "立即賣出，避免進一步損失。",
             "Sell immediately to avoid further losses.", 1),
            ("B", "賣出部分資產，降低風險。",
             "Sell part of my holdings to reduce risk.", 2),
            ("C", "先持有並觀察市場變化。",
             "Hold and monitor market conditions.", 3),
            ("D", "重新評估財務狀況與投資理由，再考慮是否加碼。",
             "Review my finances and investment rationale before considering adding more.", 4)
        ]
    },

    {
        "id": "Q6",
        "section": "risk",
        "zh": "您對投資商品與市場的熟悉程度如何？",
        "en": "How familiar are you with investment products and markets?",
        "options": [
            ("A", "幾乎不了解，也沒有投資經驗。",
             "I have little knowledge and no investment experience.", 1),
            ("B", "了解一些基本概念，經驗有限。",
             "I know some basic concepts and have limited experience.", 2),
            ("C", "了解常見投資商品及其風險。",
             "I understand common investment products and their risks.", 3),
            ("D", "熟悉多種投資商品及其風險。",
             "I am familiar with various investment products and their risks.", 4)
        ]
    },



    # -----------------------------------------------------
    # Part 2: Behavioral Characteristics
    # -----------------------------------------------------

    {
        "id": "Q7",
        "section": "behavior",
        "zh": "當市場突然大幅下跌時，您通常會有什麼感受？",
        "en": "How do you usually feel when the market suddenly drops?",
        "options": [
            ("A", "非常焦慮，難以專注其他事情。",
             "Very anxious and find it hard to focus on other things.", 4),
            ("B", "有些擔心，會頻繁查看市場。",
             "Somewhat worried and check the market frequently.", 3),
            ("C", "會在意，但仍能正常思考。",
             "Concerned, but still able to think clearly.", 2),
            ("D", "大致平靜，不會過度受到影響。",
             "Generally calm and not overly affected.", 1)
        ]
    },

    {
        "id": "Q8",
        "section": "behavior",
        "zh": "當投資出現虧損時，您通常會怎麼想？",
        "en": "How do you usually think when an investment is losing money?",
        "options": [
            ("A", "不願意實現虧損，想等到回本再賣。",
             "I avoid realizing losses and prefer to wait until I break even.", 4),
            ("B", "即使原本的投資理由改變，仍傾向繼續持有。",
             "I tend to hold even when my original investment rationale changes.", 3),
            ("C", "重新檢視投資理由與相關資訊。",
             "I review my investment rationale and relevant information.", 2),
            ("D", "依照事先設定的原則評估是否調整或退出。",
             "I evaluate whether to adjust or exit based on predefined rules.", 1)
        ]
    },

    {
        "id": "Q9",
        "section": "behavior",
        "zh": "當朋友或社群推薦熱門投資標的時，您通常會怎麼做？",
        "en": "What do you usually do when friends or social media recommend a popular investment?",
        "options": [
            ("A", "立即跟進，避免錯過機會。",
             "Follow immediately to avoid missing the opportunity.", 4),
            ("B", "容易受到影響，可能跟著買進。",
             "I am easily influenced and may buy along with others.", 3),
            ("C", "先查證資訊，再決定是否投資。",
             "I verify the information before deciding.", 2),
            ("D", "依照自己的目標與研究結果決定。",
             "I decide based on my own goals and research.", 1)
        ]
    },

    {
        "id": "Q10",
        "section": "behavior",
        "zh": "您對自己理解投資資訊與判斷風險的信心如何？",
        "en": "How confident are you in understanding investment information and assessing risk?",
        "options": [
            ("A", "沒有信心，通常需要他人協助。",
             "Not confident; I usually need help.", 1),
            ("B", "有一些信心，但仍需要協助。",
             "Somewhat confident, but I still need assistance.", 2),
            ("C", "有信心，也知道自己有哪些不足。",
             "Confident, while aware of my limitations.", 3),
            ("D", "非常有信心，能獨立判斷。",
             "Very confident and able to make independent judgments.", 4)
        ]
    },

    {
        "id": "Q11",
        "section": "behavior",
        "zh": "您認為自己成功挑選出表現優於市場標的的機率如何？",
        "en": "How likely do you think you are to select investments that outperform the market?",
        "options": [
            ("A", "不確定。",
             "I am not sure.", 1),
            ("B", "偶爾可以。",
             "Occasionally.", 2),
            ("C", "相當有機會。",
             "Quite likely.", 3),
            ("D", "大多時候都可以。",
             "Most of the time.", 4)
        ]
    },

    {
        "id": "Q12",
        "section": "behavior",
        "zh": "您平常的投資決策習慣為何？",
        "en": "What is your usual investment decision-making habit?",
        "options": [
            ("A", "常因短期消息或情緒改變決定。",
             "I often change decisions due to short-term news or emotions.", 1),
            ("B", "有時會偏離原本計畫。",
             "I sometimes deviate from my original plan.", 2),
            ("C", "通常會查證資訊並依照計畫行動。",
             "I usually verify information and follow my plan.", 3),
            ("D", "有明確原則，並定期檢視決策。",
             "I follow clear principles and review decisions regularly.", 4)
        ]
    },


    # -----------------------------------------------------
    # Part 3: Financial Capacity
    # -----------------------------------------------------

    {
        "id": "Q13",
        "section": "financial",
        "zh": "您每月的個人收入大約是多少？",
        "en": "What is your approximate monthly personal income?",
        "options": [
            ("A", "目前沒有個人收入。",
             "No personal income.", 1),
            ("B", "新臺幣 1–40,000 元。",
             "NTD 1–40,000.", 2),
            ("C", "新臺幣 40,001–80,000 元。",
             "NTD 40,001–80,000.", 3),
            ("D", "新臺幣 80,001 元以上。",
             "Above NTD 80,000.", 4)
        ],
        "sensitive": True
    },

    {
        "id": "Q14",
        "section": "financial",
        "zh": "扣除近期必要支出後，您目前有多少資金可用於投資？",
        "en": "How much money is currently available for investment after near-term essential expenses?",
        "options": [
            ("A", "目前沒有可投資資金。",
             "No money currently available for investment.", 1),
            ("B", "新臺幣 1–50,000 元。",
             "NTD 1–50,000.", 2),
            ("C", "新臺幣 50,001–300,000 元。",
             "NTD 50,001–300,000.", 3),
            ("D", "新臺幣 300,000 元以上。",
             "Above NTD 300,000.", 4)
        ],
        "sensitive": True
    },

    {
        "id": "Q15",
        "section": "financial",
        "zh": "您每月必要生活支出與還款大約是多少？",
        "en": "What are your approximate monthly essential living expenses and debt repayments?",
        "options": [
            ("A", "新臺幣 0–10,000 元。",
             "NTD 0–10,000.", 4),
            ("B", "新臺幣 10,001–25,000 元。",
             "NTD 10,001–25,000.", 3),
            ("C", "新臺幣 25,001–50,000 元。",
             "NTD 25,001–50,000.", 2),
            ("D", "新臺幣 50,000 元以上。",
             "Above NTD 50,000.", 1)
        ],
        "sensitive": True
    },

    {
        "id": "Q16",
        "section": "financial",
        "zh": "您目前的總負債大約是多少？",
        "en": "What is your approximate total debt?",
        "options": [
            ("A", "目前沒有負債。",
             "No debt.", 4),
            ("B", "新臺幣 1–100,000 元。",
             "NTD 1–100,000.", 3),
            ("C", "新臺幣 100,001–500,000 元。",
             "NTD 100,001–500,000.", 2),
            ("D", "新臺幣 500,000 元以上。",
             "Above NTD 500,000.", 1)
        ],
        "sensitive": True
    },

    {
        "id": "Q17",
        "section": "financial",
        "zh": "如果突然需要支付新臺幣 30,000 元的必要支出，您能如何應付？",
        "en": "If you suddenly needed to pay NTD 30,000 in essential expenses, how would you manage?",
        "options": [
            ("A", "無法應付，需要額外資金。",
             "I could not manage without additional funds.", 1),
            ("B", "可以應付，但需要借款或延後其他支出。",
             "I could manage by borrowing or postponing other expenses.", 2),
            ("C", "可以用現有資金支付，但可用資金會明顯減少。",
             "I could pay with existing funds, but my available funds would decrease significantly.", 3),
            ("D", "可以應付，且不會明顯影響原本財務安排。",
             "I could manage without significantly affecting my financial plans.", 4)
        ],
        "sensitive": True
    },

    {
    "id": "Q18",
    "section": "financial",
    "zh": "您目前的可投資資金，多久之內不需要拿來支付生活支出？",
    "en": "How long can your investment funds remain untouched without being needed for living expenses?",
    "options": [
        ("A", "不到一年。", "Less than 1 year.", 1),
        ("B", "超過 1 年至 3 年。", "More than 1 year, up to 3 years.", 2),
        ("C", "超過 3 年至 5 年。", "More than 3 years, up to 5 years.", 3),
        ("D", "超過 5 年。", "More than 5 years.", 4)
    ]
},   # ← Q18 結尾要逗號，因為還有 Q19

{
    "id": "Q19",
    "section": "risk",
    "zh": "這筆資金主要用途是？",
    "en": "What is the primary purpose of this investment?",
    "options": [
        ("A", "緊急預備金", "Emergency reserve", 1),
        ("B", "買房頭期款", "Home down payment", 2),
        ("C", "子女教育基金", "Education fund", 3),
        ("D", "退休資產", "Retirement asset", 4),
        ("E", "其他", "Other", 2)
    ]
},   # ← Q19 結尾要逗號，因為還有 Q20

{
    "id": "Q20",
    "section": "behavior",
    "zh": "如果朋友投資報酬率比你高很多，你會怎麼反應？",
    "en": "If your friends' investment returns are much higher than yours, how would you react?",
    "options": [
        ("A", "保持原有策略，不受影響。", "Stick to my strategy, unaffected.", 1),
        ("B", "考慮調整投資方式。", "Consider adjusting my investments.", 2),
        ("C", "感到焦慮，想追趕報酬。", "Feel anxious and want to catch up.", 3),
        ("D", "覺得自己投資失敗。", "Feel like my investment failed.", 4)
    ]
}   
]

# =========================================================
# 4. 問卷區塊名稱
# =========================================================

SECTIONS = {
    "risk": {
        "zh": "第一部分｜傳統風險偏好",
        "en": "Part 1 | Traditional Risk Profiling",
        "description_zh": "了解您對投資風險、報酬與資金使用時間的看法。",
        "description_en": "Understand your views on investment risk, returns, and time horizon."
    },
    "behavior": {
        "zh": "第二部分｜行為與情緒特徵",
        "en": "Part 2 | Behavioral Characteristics",
        "description_zh": "了解您面對市場變化時的情緒、判斷與決策習慣。",
        "description_en": "Understand your emotions, judgments, and decision habits in changing markets."
    },
    "financial": {
        "zh": "第三部分｜財務承受能力",
        "en": "Part 3 | Financial Capacity",
        "description_zh": "了解您目前的資金彈性與財務條件。",
        "description_en": "Understand your current financial flexibility and capacity."
    }
}


# =========================================================
# 5. 顯示雙語題目
# =========================================================

def show_question(question):

    st.markdown(
        f'<div class="question-zh">'
        f'{question["id"]}. {question["zh"]}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="question-en">'
        f'{question["en"]}'
        f'</div>',
        unsafe_allow_html=True
    )

    option_dict = {
        option[0]: option
        for option in question["options"]
    }

    selected = st.radio(
        label=f'請選擇最符合您的答案｜Select your answer',
        options=list(option_dict.keys()),
        format_func=lambda key: (
            f'{option_dict[key][1]}\n'
            f'{option_dict[key][2]}'
        ),
        index=None,
        key=question["id"],
        label_visibility="collapsed"
    )

    if selected is None:
        return None

    return option_dict[selected][3]


# =========================================================
# 6. 計分函式
# =========================================================

def calculate_normalized_score(score, min_score, max_score):
    """
    將原始分數轉換為 0–100 分。
    """
    if score is None:
        return None

    return round(
        (score - min_score) / (max_score - min_score) * 100,
        1
    )


def classify_risk(score):
    """
    Traditional Risk Profiling:
    Q1–Q5，總分 5–20。
    """
    if score is None:
        return "Insufficient Data"

    normalized = calculate_normalized_score(score, 5, 20)

    if normalized <= 33.3:
        return "Conservative"
    elif normalized <= 66.6:
        return "Moderate"
    else:
        return "Aggressive"


def classify_financial(score, answered_count):
    """
    Financial Capacity:
    Q13–Q18，總分 6–24。
    若有效回答不足 4 題，不進行分類。
    """
    if answered_count < 4:
        return "Insufficient Data"

    normalized = calculate_normalized_score(
        score,
        answered_count,
        answered_count * 4
    )

    if normalized <= 33.3:
        return "Limited"
    elif normalized <= 66.6:
        return "Moderate"
    else:
        return "Relatively Strong"


def get_behavior_label(question_id, score):
    """
    將行為分數轉換為文字標籤。
    """
    if score is None:
        return "Not answered"

    if question_id in ["Q7", "Q8", "Q9"]:
        if score >= 3:
            return "Higher tendency"
        else:
            return "Lower tendency"

    elif question_id in ["Q10", "Q11"]:
        if score >= 3:
            return "Higher perceived confidence"
        else:
            return "Lower perceived confidence"

    elif question_id == "Q12":
        if score >= 3:
            return "Higher decision discipline"
        else:
            return "Lower decision discipline"

    return "Not classified"


# =========================================================
# 7. 產生 AI Profile 的基礎文字
# =========================================================

def generate_profile(risk_type, financial_type, behavior_results):

    paragraphs = []

    # 風險偏好
    if risk_type == "Conservative":
        paragraphs.append(
            "您的問卷回答呈現較保守的風險偏好，"
            "您較重視本金安全與降低投資波動。"
        )

    elif risk_type == "Moderate":
        paragraphs.append(
            "您的問卷回答呈現中等風險偏好，"
            "您在資產成長與風險控制之間有所平衡。"
        )

    elif risk_type == "Aggressive":
        paragraphs.append(
            "您的問卷回答呈現較積極的風險偏好，"
            "您對投資波動與較高風險的接受程度較高。"
        )

    else:
        paragraphs.append(
            "目前風險偏好資料不足，無法產生完整分類。"
        )

    # 財務承受能力
    if financial_type == "Limited":
        paragraphs.append(
            "財務承受能力指標顯示，您目前的財務彈性可能較有限。"
            "請留意生活支出、緊急預備金與投資資金的區隔。"
        )

    elif financial_type == "Moderate":
        paragraphs.append(
            "您的財務承受能力指標落在中間區間，"
            "仍需依實際支出與資金需求評估可承擔的風險。"
        )

    elif financial_type == "Relatively Strong":
        paragraphs.append(
            "您的財務承受能力指標相對較高，"
            "但這不代表應承擔更高投資風險。"
        )

    else:
        paragraphs.append(
            "財務資料不足，暫時無法完成財務承受能力分類。"
        )

    # 行為特徵
    anxiety = behavior_results.get("Q7")
    loss_aversion = behavior_results.get("Q8")
    herding = behavior_results.get("Q9")
    discipline = behavior_results.get("Q12")
    confidence = behavior_results.get("Q11")

    if anxiety is not None and anxiety >= 3:
        paragraphs.append(
            "您在市場下跌時可能較容易感到焦慮，"
            "可以在做決策前先確認原本的投資目標與計畫。"
        )

    if loss_aversion is not None and loss_aversion >= 3:
        paragraphs.append(
            "您的回答呈現較明顯的損失規避傾向，"
            "面對虧損時可能較難依照原先原則調整決策。"
        )

    if herding is not None and herding >= 3:
        paragraphs.append(
            "您的回答呈現較明顯的從眾傾向，"
            "接收熱門投資資訊時，可以先查證來源與風險。"
        )

    if confidence is not None and confidence >= 3:
        paragraphs.append(
            "您對自己挑選優於市場標的的能力具有較高主觀信心，"
            "但自我評估不等於實際投資績效。"
        )

    if discipline is not None and discipline >= 3:
        paragraphs.append(
            "您的回答顯示較有依照計畫進行投資決策的傾向。"
        )

    return "\n\n".join(paragraphs)

# =========================================================
# 9. 分頁式問卷表單
# =========================================================

# 初始化狀態
if "index" not in st.session_state:
    st.session_state.index = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "submitted" not in st.session_state:
    st.session_state.submitted = False

total_questions = len(QUESTIONS)
current_index = st.session_state.index
current_question = QUESTIONS[current_index]

# 顯示進度條
progress_value = int(current_index / total_questions * 100)
st.progress(progress_value, text=f"已完成 {current_index}/{total_questions} 題")

# 顯示題目
st.markdown(
    f'<div class="question-zh">{current_question["id"]}. {current_question["zh"]}</div>',
    unsafe_allow_html=True
)
st.markdown(
    f'<div class="question-en">{current_question["en"]}</div>',
    unsafe_allow_html=True
)

option_dict = {opt[0]: opt for opt in current_question["options"]}
choice = st.radio(
    "請選擇最符合您的答案｜Select your answer",
    options=list(option_dict.keys()),
    format_func=lambda key: f'{option_dict[key][1]}\n{option_dict[key][2]}',
    index=None,
    key=current_question["id"]
)

# 提交答案
if st.button("提交答案"):
    if choice is not None:
        st.session_state.answers[current_question["id"]] = option_dict[choice][3]

        # 如果還有下一題 → 跳到下一題
        if current_index < total_questions - 1:
            st.session_state.index += 1
            st.rerun()   # ✅ 新版正確用法
        else:
            st.success("🎉 全部完成！")
            st.session_state.submitted = True
    else:
        st.warning("請先選擇一個答案再提交。")

# =========================================================
# 10. 按下分析後計算結果 + JSON 輸出 + 顯示分析結果
# =========================================================

if st.session_state.submitted:
    # -----------------------------------------------------
    # 風險偏好 Q1–Q5
    # -----------------------------------------------------
    risk_ids = ["Q1", "Q2", "Q3", "Q4", "Q5"]

    risk_scores = [
        st.session_state.answers.get(qid)
        for qid in risk_ids
        if st.session_state.answers.get(qid) is not None
    ]

    risk_answered_count = len(risk_scores)

    if risk_answered_count == 5:
        risk_raw_score = sum(risk_scores)
        risk_normalized_score = calculate_normalized_score(risk_raw_score, 5, 20)
        risk_type = classify_risk(risk_raw_score)
    else:
        risk_raw_score = None
        risk_normalized_score = None
        risk_type = "Insufficient Data"

    # -----------------------------------------------------
    # 投資熟悉度 Q6 (正規化到 0–100)
    # -----------------------------------------------------
    familiarity_raw = st.session_state.answers.get("Q6")

    if familiarity_raw is not None:
        familiarity_score = calculate_normalized_score(familiarity_raw, 1, 4)
    else:
        familiarity_score = None

    # -----------------------------------------------------
    # 行為特徵 Q7–Q12
    # -----------------------------------------------------
    behavior_ids = ["Q7", "Q8", "Q9", "Q10", "Q11", "Q12"]
    behavior_results = {qid: st.session_state.answers.get(qid) for qid in behavior_ids}

    # -----------------------------------------------------
    # 財務承受能力 Q13–Q18
    # -----------------------------------------------------
    financial_ids = ["Q13", "Q14", "Q15", "Q16", "Q17", "Q18"]

    financial_scores = [
        st.session_state.answers.get(qid)
        for qid in financial_ids
        if st.session_state.answers.get(qid) is not None
    ]

    financial_answered_count = len(financial_scores)

    if financial_answered_count > 0:
        financial_raw_score = sum(financial_scores)
        financial_normalized_score = calculate_normalized_score(
            financial_raw_score,
            financial_answered_count,
            financial_answered_count * 4
        )
        financial_type = classify_financial(financial_raw_score, financial_answered_count)
    else:
        financial_raw_score = None
        financial_normalized_score = None
        financial_type = "Insufficient Data"

    # -----------------------------------------------------
    # Investor Type
    # -----------------------------------------------------
    if risk_type == "Insufficient Data":
        investor_type = "Insufficient Data"
    else:
        investor_type = f"{risk_type} Investor"

    # 中英文對照表
    investor_type_map = {
        "Conservative Investor": "保守型投資者",
        "Moderate Investor": "穩健型投資者",
        "Aggressive Investor": "積極型投資者",
        "Insufficient Data": "資料不足"
    }

    investor_type_zh = investor_type_map.get(investor_type, "未知類型")
    st.subheader("Investor Type｜投資者類型")
    st.success(f"{investor_type}｜{investor_type_zh}")

    # -----------------------------------------------------
    # AI Profile（規則式文字）
    # -----------------------------------------------------
    profile_text = generate_profile(risk_type, financial_type, behavior_results)

    # -----------------------------------------------------
    # JSON 輸出（不顯示，只提供下載）
    # -----------------------------------------------------
    json_data = {
        "profile_version": "1.0",
        "language": "zh-TW",
        "investor_type": investor_type,
        "traditional_risk": {
            "raw_score": risk_raw_score,
            "max_score": 20,
            "normalized_score": risk_normalized_score,
            "classification": risk_type,
            "answered_count": risk_answered_count,
            "total_questions": 5
        },
        "investment_familiarity": {
            "q6_score": familiarity_score
        },
        "behavioral_characteristics": behavior_results,
        "financial_capacity": {
            "raw_score": financial_raw_score,
            "max_score": financial_answered_count * 4,
            "normalized_score": financial_normalized_score,
            "classification": financial_type,
            "answered_count": financial_answered_count,
            "total_questions": 6
        },
        "profile_summary": profile_text
    }

    json_string = json.dumps(json_data, ensure_ascii=False, indent=4)

    st.download_button(
        label="下載 JSON 資料｜Download JSON",
        data=json_string,
        file_name="investor_profile.json",
        mime="application/json"
    )

    # =====================================================
    # 11. 顯示分析結果
    # =====================================================
    st.divider()
    st.markdown('<div class="section-title-zh">您的分析結果</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title-en">Your Investor Profile</div>', unsafe_allow_html=True)

    st.subheader("Investor Type｜投資者類型")
    st.success(investor_type)

    st.write("此類型主要根據 Q1–Q5 的風險偏好回答產生，財務能力與行為特徵會另外呈現。")

    # 分數卡片
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("風險偏好分數｜Risk Preference",
                  f"{risk_normalized_score}/100" if risk_normalized_score is not None else "資料不足")
    with col2:
        st.metric("財務承受能力｜Financial Capacity",
                  f"{financial_normalized_score}/100" if financial_normalized_score is not None else "資料不足")
    with col3:
        st.metric("投資熟悉度｜Familiarity",
                  f"{familiarity_score}/100" if familiarity_score is not None else "資料不足")

    # -----------------------------------------------------
    # 雷達圖（中文大字 + 英文小字）
    # -----------------------------------------------------
    import plotly.graph_objects as go
    categories = ["Risk Preference｜風險偏好", "Financial Capacity｜財務能力", "Investment Familiarity｜投資熟悉度"]
    values = [
        risk_normalized_score if risk_normalized_score is not None else 0,
        financial_normalized_score if financial_normalized_score is not None else 0,
        familiarity_score if familiarity_score is not None else 0
    ]

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Investor Profile'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        title="投資者輪廓雷達圖｜Investor Profile Radar Chart"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Profile 文字敘述
    st.write(profile_text)
    # =====================================================
    # 交集圖：風險態度、財務能力、投資熟悉度
    # =====================================================
    st.subheader("風險交集圖｜Risk Attitude, Perception, and Capacity")

    if risk_normalized_score is not None and financial_normalized_score is not None and familiarity_score is not None:
        intersection_score = min(risk_normalized_score, financial_normalized_score, familiarity_score)

        fig = go.Figure()

        fig.add_shape(type="circle", xref="x", yref="y",
                  x0=0, y0=0, x1=risk_normalized_score/25, y1=risk_normalized_score/25,
                  line_color="rgba(0,0,255,0.3)", fillcolor="rgba(0,0,255,0.2)")
        fig.add_shape(type="circle", xref="x", yref="y",
                  x0=1, y0=0, x1=1+financial_normalized_score/25, y1=financial_normalized_score/25,
                  line_color="rgba(255,0,0,0.3)", fillcolor="rgba(255,0,0,0.2)")
        fig.add_shape(type="circle", xref="x", yref="y",
                  x0=0.5, y0=1, x1=0.5+familiarity_score/25, y1=1+familiarity_score/25,
                  line_color="rgba(0,255,0,0.3)", fillcolor="rgba(0,255,0,0.2)")

    # 標籤文字
        fig.add_annotation(x=0.3, y=1.8, text="Attitude｜心理意願", showarrow=False, font=dict(size=14))
        fig.add_annotation(x=2.2, y=1.8, text="Capacity｜財務能力", showarrow=False, font=dict(size=14))
        fig.add_annotation(x=1.3, y=0.3, text="Perception｜投資熟悉度", showarrow=False, font=dict(size=14))
        fig.add_annotation(x=1.3, y=1.3,
                       text=f"適合範圍 Intersection: {intersection_score}%",
                       showarrow=False, font=dict(size=16, color="black"))

        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(height=400, width=600, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # 說明文字 + 舉例
        st.markdown(f"""
        ### 交集圖解釋
        這張圖顯示三個面向：
        - **心理意願 (Attitude)**：Q1–Q5 的風險偏好。
        - **財務能力 (Capacity)**：Q13–Q18 的承受能力。
        - **投資熟悉度 (Perception)**：Q6 的熟悉度。

        三者的交集才是最適合的投資範圍，不是分數最高就最好。  
        👉 目前你的交集分數為 **{intersection_score}%**。

        #### 舉例情境：
        - **例子 1**：心理意願高（想追高報酬），但財務能力低（收入有限、負債高）。  
        → 圖的交集很小，提醒你「雖然想冒險，但財務條件不支持」。  

        - **例子 2**：財務能力強（資金充足），但心理意願低（怕波動）。  
        → 圖的交集也小，提醒你「雖然能承受，但心裡不願意」。  

        - **例子 3**：三者都中等。  
        → 圖的交集大，代表「比較平衡，適合穩健投資」。  
        """)
    else:
        st.warning("⚠️ 三個分數資料不足，無法生成交集圖。")

# =========================================================
# 12. 頁尾
# =========================================================
st.divider()
st.caption("Digital Investor Profiling System | Academic Project")