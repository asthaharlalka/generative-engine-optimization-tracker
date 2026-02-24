import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI
import requests
import json
import re
from collections import defaultdict

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GEO Visibility Tracker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

* { font-family: 'Syne', sans-serif; }
code, .mono { font-family: 'DM Mono', monospace; }

.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

.hero {
    padding: 3rem 0 2rem 0;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 2.5rem;
}

.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: #7c3aed;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.05;
    color: #f0f0fa;
    margin-bottom: 0.75rem;
}

.hero-title span {
    color: #7c3aed;
}

.hero-sub {
    font-size: 1rem;
    color: #888;
    font-weight: 400;
    max-width: 560px;
}

.stat-pill {
    display: inline-block;
    background: #13131f;
    border: 1px solid #2a2a3e;
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #7c3aed;
    margin-right: 0.5rem;
    margin-top: 0.75rem;
}

.input-section {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.75rem;
    margin-bottom: 1.5rem;
}

.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    color: #555;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* Score cards */
.score-card {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}

.score-card:hover { border-color: #7c3aed; }

.score-number {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.25rem;
}

.score-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: #666;
    text-transform: uppercase;
}

.score-brand {
    font-size: 0.9rem;
    font-weight: 600;
    color: #e8e8f0;
    margin-top: 0.5rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Winner / loser callouts */
.winner-box {
    background: linear-gradient(135deg, #0d1f0d, #0f1a0f);
    border: 1px solid #1a4a1a;
    border-left: 3px solid #22c55e;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}

.loser-box {
    background: linear-gradient(135deg, #1f0d0d, #1a0f0f);
    border: 1px solid #4a1a1a;
    border-left: 3px solid #ef4444;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}

.callout-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

.callout-text {
    font-size: 0.92rem;
    color: #e8e8f0;
}

/* Question log */
.question-row {
    background: #0a0a14;
    border: 1px solid #15152a;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    color: #aaa;
}

.question-row strong {
    color: #e8e8f0;
    font-weight: 600;
}

/* Sticker badge */
.geo-badge {
    display: inline-block;
    background: #7c3aed;
    color: white;
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    margin-left: 0.5rem;
    vertical-align: middle;
}

/* Streamlit overrides */
.stButton > button {
    background: #7c3aed;
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.03em;
    padding: 0.65rem 2rem;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover { background: #6d28d9; }

.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: #13131f !important;
    border: 1px solid #2a2a3e !important;
    color: #e8e8f0 !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
}

.stTextInput > label, .stSelectbox > label {
    color: #888 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

div[data-testid="stExpander"] {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
}

.stSpinner > div { color: #7c3aed !important; }

hr { border-color: #1e1e2e !important; margin: 2rem 0 !important; }

/* Hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; max-width: 1100px; }
</style>
""", unsafe_allow_html=True)


# ── Data: 10 buyer questions per category ────────────────────────────────────
BUYER_QUESTIONS = {
    "CRM Software": [
        "What's the best CRM for a 50-person B2B sales team?",
        "Which CRM has the best Salesforce alternative for mid-market?",
        "What CRM do fast-growing SaaS companies use?",
        "Best CRM software with built-in AI features in 2025?",
        "Which CRM is easiest to set up without a Salesforce admin?",
        "What CRM integrates best with HubSpot marketing?",
        "CRM with best pipeline visibility and forecasting?",
        "Which CRM do SDR teams prefer?",
        "Best CRM for a company switching from spreadsheets?",
        "Which CRM has the best customer support and onboarding?"
    ],
    "Marketing Automation": [
        "What's the best marketing automation platform for B2B SaaS?",
        "Best alternative to Marketo for a mid-market company?",
        "Which marketing automation tool has the best email deliverability?",
        "What marketing automation platforms do growth teams use?",
        "Best marketing automation for lead nurturing in 2025?",
        "Which platform is better: HubSpot or Pardot for B2B?",
        "Marketing automation with best CRM integration?",
        "What's the easiest marketing automation tool to set up?",
        "Best marketing automation for ABM campaigns?",
        "Which marketing automation platform has the best analytics?"
    ],
    "Project Management": [
        "What's the best project management tool for engineering teams?",
        "Best alternative to Jira for a non-technical team?",
        "Which project management software do startups prefer?",
        "Best project management tool with AI features in 2025?",
        "What project management tool works best with Slack?",
        "Asana vs Monday.com vs Notion — which is best?",
        "Best project management software for remote teams?",
        "Which PM tool has the best reporting and dashboards?",
        "Best project management tool for agencies?",
        "What PM software scales from 10 to 200 employees?"
    ],
    "HR Software": [
        "Best HRIS for a company with 100-500 employees?",
        "What HR software do fast-growing startups use?",
        "Best alternative to Workday for mid-market?",
        "Which HR platform has the best onboarding features?",
        "Best HR software with payroll integration in 2025?",
        "What HRIS has the best employee self-service?",
        "Which HR platform is easiest to implement?",
        "Best HR software for performance management?",
        "What HR tool integrates best with Slack and GSuite?",
        "Which HRIS has the best analytics and reporting?"
    ],
    "Data Analytics": [
        "Best BI tool for a data team without engineers?",
        "What analytics platform do SaaS companies use for product metrics?",
        "Best alternative to Tableau for a startup?",
        "Which BI tool has the best self-service analytics?",
        "Looker vs Power BI vs Metabase — which is best?",
        "Best analytics tool that connects to Snowflake?",
        "What BI platform do product managers prefer?",
        "Best data visualization tool for non-technical users?",
        "Which analytics tool has the best dashboard sharing?",
        "Best BI tool for a company scaling to enterprise?"
    ]
}

# ── Helper: call OpenAI ───────────────────────────────────────────────────────
def query_openai(question: str, client: OpenAI) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": question}],
            max_tokens=400,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"


# ── Helper: call Perplexity ───────────────────────────────────────────────────
def query_perplexity(question: str, api_key: str) -> str:
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "sonar",
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 400,
            "temperature": 0.3
        }
        r = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=20
        )
        data = r.json()
        if "choices" not in data:
            return f"ERROR: Perplexity returned — {data.get('error', {}).get('message', str(data))}"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {str(e)}"


# ── Helper: call Anthropic Claude ────────────────────────────────────────────
def query_claude(question: str, api_key: str) -> str:
    try:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 400,
            "messages": [{"role": "user", "content": question}]
        }
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=15
        )
        return r.json()["content"][0]["text"]
    except Exception as e:
        return f"ERROR: {str(e)}"


# ── Helper: call Google Gemini ────────────────────────────────────────────────
def query_gemini(question: str, api_key: str) -> str:
    for model in ["gemini-2.0-flash", "gemini-1.5-flash"]:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": question}]}],
                "generationConfig": {"maxOutputTokens": 400, "temperature": 0.3}
            }
            r = requests.post(url, json=payload, timeout=20)
            data = r.json()
            if "candidates" in data:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            # If this model failed, try the next one
            error_msg = data.get("error", {}).get("message", "")
            if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                continue
            return f"ERROR: Gemini — {error_msg or str(data)}"
        except Exception as e:
            return f"ERROR: {str(e)}"
    return "ERROR: Gemini — no available model worked for this API key"


# ── Helper: call Grok (xAI) ───────────────────────────────────────────────────
def query_grok(question: str, api_key: str) -> str:
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "grok-beta",
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 400,
            "temperature": 0.3
        }
        r = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {str(e)}"


# ── Helper: count brand mentions in a response ────────────────────────────────
def count_mentions(text: str, brands: list[str]) -> dict:
    text_lower = text.lower()
    counts = {}
    for brand in brands:
        pattern = re.compile(re.escape(brand.lower()))
        counts[brand] = len(pattern.findall(text_lower))
    return counts


# ── Helper: score sentiment (naive but fast) ──────────────────────────────────
POSITIVE_WORDS = ["best", "great", "excellent", "recommend", "top", "leading",
                  "popular", "preferred", "powerful", "easy", "intuitive", "love"]
NEGATIVE_WORDS = ["avoid", "bad", "poor", "slow", "expensive", "difficult",
                  "complicated", "overpriced", "bloated", "legacy", "outdated"]

def sentiment_in_context(text: str, brand: str) -> str:
    text_lower = text.lower()
    idx = text_lower.find(brand.lower())
    if idx == -1:
        return "neutral"
    context = text_lower[max(0, idx-100):idx+150]
    pos = sum(1 for w in POSITIVE_WORDS if w in context)
    neg = sum(1 for w in NEGATIVE_WORDS if w in context)
    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    return "neutral"


# ── Scorecard builder ─────────────────────────────────────────────────────────
def build_scorecard(results: list[dict], brands: list[str], llm_keys: list[str]) -> pd.DataFrame:
    total_q = len(results)
    brand_data = defaultdict(lambda: {
        "questions_appeared": 0,
        "sentiment_positive": 0,
        "total_mentions": 0,
        **{f"mentions_{llm}": 0 for llm in llm_keys}
    })

    for r in results:
        appeared = defaultdict(bool)
        for i, llm in enumerate(llm_keys):
            m = count_mentions(r[f"{llm}_response"], brands)
            for brand in brands:
                if m[brand] > 0:
                    brand_data[brand][f"mentions_{llm}"] += m[brand]
                    brand_data[brand]["total_mentions"] += m[brand]
                    appeared[brand] = True
                    if i == 0:
                        sent = sentiment_in_context(r[f"{llm}_response"], brand)
                        brand_data[brand]["sentiment_positive"] += (1 if sent == "positive" else 0)
        for brand in brands:
            if appeared[brand]:
                brand_data[brand]["questions_appeared"] += 1

    rows = []
    for brand in brands:
        d = brand_data[brand]
        visibility_score = round(
            (d["questions_appeared"] / total_q * 60) +
            (min(d["total_mentions"], 20 * len(llm_keys)) / (20 * len(llm_keys)) * 25) +
            (d["sentiment_positive"] / max(1, d["questions_appeared"]) * 15)
        )
        row = {
            "Brand": brand,
            "Visibility Score": visibility_score,
            "Questions Appeared In": d["questions_appeared"],
            "Total Questions": total_q,
            "Total Mentions": d["total_mentions"],
            "AI Recommendation Risk": "🔴 Invisible" if d["questions_appeared"] == 0
                                      else "🟡 Weak" if d["questions_appeared"] <= 3
                                      else "🟢 Visible"
        }
        for llm in llm_keys:
            row[f"{llm.title()} Mentions"] = d[f"mentions_{llm}"]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Visibility Score", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


# ── Plotly bar chart ──────────────────────────────────────────────────────────
def make_bar_chart(df: pd.DataFrame) -> go.Figure:
    colors = []
    for score in df["Visibility Score"]:
        if score >= 60:
            colors.append("#22c55e")
        elif score >= 30:
            colors.append("#f59e0b")
        else:
            colors.append("#ef4444")

    fig = go.Figure(go.Bar(
        x=df["Brand"],
        y=df["Visibility Score"],
        marker_color=colors,
        marker_line_width=0,
        text=df["Visibility Score"],
        textposition="outside",
        textfont=dict(family="DM Mono", size=13, color="#e8e8f0")
    ))

    fig.update_layout(
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0a0a0f",
        font=dict(family="Syne", color="#888"),
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickfont=dict(family="Syne", size=13, color="#e8e8f0")
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#1e1e2e", zeroline=False,
            range=[0, 110],
            tickfont=dict(family="DM Mono", size=11, color="#555")
        ),
        margin=dict(t=30, b=20, l=20, r=20),
        height=320,
        bargap=0.35,
        showlegend=False
    )
    return fig


def make_mention_comparison(df: pd.DataFrame, llm_keys: list[str]) -> go.Figure:
    LLM_COLORS = {
        "gpt": "#7c3aed",
        "perplexity": "#06b6d4",
        "claude": "#f59e0b",
        "gemini": "#22c55e",
        "grok": "#f87171"
    }
    LLM_LABELS = {
        "gpt": "ChatGPT",
        "perplexity": "Perplexity",
        "claude": "Claude",
        "gemini": "Gemini",
        "grok": "Grok"
    }
    fig = go.Figure()
    for llm in llm_keys:
        col = f"{llm.title()} Mentions"
        if col in df.columns:
            fig.add_trace(go.Bar(
                name=LLM_LABELS.get(llm, llm.title()),
                x=df["Brand"],
                y=df[col],
                marker_color=LLM_COLORS.get(llm, "#888"),
                marker_line_width=0,
            ))
    fig.update_layout(
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0a0a0f",
        barmode="group",
        font=dict(family="Syne", color="#888"),
        xaxis=dict(showgrid=False, zeroline=False,
                   tickfont=dict(family="Syne", size=12, color="#e8e8f0")),
        yaxis=dict(showgrid=True, gridcolor="#1e1e2e", zeroline=False,
                   tickfont=dict(family="DM Mono", size=11, color="#555")),
        legend=dict(
            font=dict(family="DM Mono", size=11, color="#888"),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom", y=1.02
        ),
        margin=dict(t=40, b=20, l=20, r=20),
        height=300,
        bargap=0.25, bargroupgap=0.08
    )
    return fig


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Generative Engine Optimization</div>
    <div class="hero-title">AI Search<br><span>Visibility Tracker</span></div>
    <div class="hero-sub">
        Find out which brands AI recommends when your buyers ask category questions —
        before your competitors do.
    </div>
    <div>
        <span class="stat-pill">94% of B2B buyers use AI to research vendors</span>
        <span class="stat-pill">Works for any category — software, FMCG, retail, services</span>
        <span class="stat-pill">Up to 5 LLMs simultaneously</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar / inputs ──────────────────────────────────────────────────────────
# Initialize session state for API keys on first load only
for key in ["openai_key", "perp_key", "claude_key", "gemini_key", "grok_key"]:
    if key not in st.session_state:
        st.session_state[key] = ""

with st.sidebar:
    st.markdown("### 🔑 API Keys")
    st.markdown('<div style="font-family: DM Mono; font-size: 0.68rem; color: #555; margin-bottom: 0.5rem;">Add any combination — scan runs with whatever you have. Keys stay for this tab session.</div>', unsafe_allow_html=True)

    st.session_state.openai_key = st.text_input("OpenAI (ChatGPT)", value=st.session_state.openai_key, type="password", placeholder="sk-...")
    st.session_state.perp_key = st.text_input("Perplexity", value=st.session_state.perp_key, type="password", placeholder="pplx-...")
    st.session_state.claude_key = st.text_input("Anthropic (Claude)", value=st.session_state.claude_key, type="password", placeholder="sk-ant-...")
    st.session_state.gemini_key = st.text_input("Google (Gemini)", value=st.session_state.gemini_key, type="password", placeholder="AIza...")
    st.session_state.grok_key = st.text_input("xAI (Grok)", value=st.session_state.grok_key, type="password", placeholder="xai-...")

    # Expose as local vars for rest of app
    openai_key = st.session_state.openai_key
    perp_key = st.session_state.perp_key
    claude_key = st.session_state.claude_key
    gemini_key = st.session_state.gemini_key
    grok_key = st.session_state.grok_key

    st.markdown("---")
    active_llms = sum([bool(openai_key), bool(perp_key), bool(claude_key), bool(gemini_key), bool(grok_key)])
    st.markdown(f"""
    <div style='font-family: DM Mono; font-size: 0.7rem; color: #555; line-height: 1.6;'>
    Keys are never stored.<br>
    Cleared when tab is closed.<br><br>
    <b style='color:#7c3aed'>{active_llms}/5 LLMs active</b>
    </div>
    """, unsafe_allow_html=True)

# ── Main inputs ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="section-label">Step 1 — Your category (type anything)</div>', unsafe_allow_html=True)
    category = st.text_input(
        "Category",
        value="CRM Software for Small Business",
        label_visibility="collapsed",
        placeholder="e.g. GTM Automation, Running Shoes, Specialty Coffee..."
    )

with col2:
    st.markdown('<div class="section-label">Step 2 — Your brand <span style="color:#ef4444">*required</span></div>', unsafe_allow_html=True)
    your_brand = st.text_input(
        "Your brand",
        value="Pipedrive",
        label_visibility="collapsed",
        placeholder="The brand you want the analysis for"
    )

st.markdown('<div class="section-label" style="margin-top:1rem;">Step 3 — Competitor brands (comma-separated)</div>', unsafe_allow_html=True)
brands_input = st.text_input(
    "Competitor brands",
    value="Salesforce, HubSpot, Zoho CRM, Monday CRM, Freshsales",
    label_visibility="collapsed",
    placeholder="Brand A, Brand B, Brand C..."
)

# Combine your brand + competitors, your brand first
competitor_list = [b.strip() for b in brands_input.split(",") if b.strip()]
brands = ([your_brand.strip()] + competitor_list) if your_brand.strip() else competitor_list

st.markdown('<div class="section-label" style="margin-top:1rem;">Step 4 — Custom questions (optional)</div>', unsafe_allow_html=True)
st.markdown(
    '<div style="font-family: DM Mono; font-size: 0.68rem; color: #444; margin-bottom: 0.5rem;">'
    'Add your own questions — one per line. These run alongside the AI-generated ones. '
    'Leave blank to use only AI-generated questions.'
    '</div>',
    unsafe_allow_html=True
)
custom_questions_raw = st.text_area(
    "Custom questions",
    value="",
    label_visibility="collapsed",
    placeholder="Which CRM is easiest to set up without a consultant?\nBest CRM for a sales team under 10 people?\nWhat CRM do founders actually recommend?",
    height=120
)
custom_questions = [q.strip() for q in custom_questions_raw.strip().split("\n") if q.strip()]

if custom_questions:
    st.markdown(
        f'<div style="font-family: DM Mono; font-size: 0.68rem; color: #7c3aed; margin-top: 0.3rem;">'
        f'✓ {len(custom_questions)} custom question{"s" if len(custom_questions) > 1 else ""} added — '
        f'will run alongside AI-generated questions.'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Run button ────────────────────────────────────────────────────────────────
run_col, _ = st.columns([1, 2])
with run_col:
    run = st.button("🔍  Run GEO Scan", use_container_width=True)

# ── Execution ─────────────────────────────────────────────────────────────────
if run:
    # Build active LLM map
    llm_map = {}
    if openai_key:
        llm_map["gpt"] = {"fn": query_openai, "args": [OpenAI(api_key=openai_key)], "key_type": "client"}
    if perp_key:
        llm_map["perplexity"] = {"fn": query_perplexity, "args": [perp_key]}
    if claude_key:
        llm_map["claude"] = {"fn": query_claude, "args": [claude_key]}
    if gemini_key:
        llm_map["gemini"] = {"fn": query_gemini, "args": [gemini_key]}
    if grok_key:
        llm_map["grok"] = {"fn": query_grok, "args": [grok_key]}

    if not llm_map:
        st.error("Add at least one API key in the sidebar to run a scan.")
        st.stop()

    if not your_brand.strip():
        st.error("Your brand name is required — it's what the analysis is built around. Add it in Step 2.")
        st.stop()

    if len(brands) < 2:
        st.error("Enter at least one competitor brand to compare against.")
        st.stop()

    llm_keys = list(llm_map.keys())
    llm_labels = {"gpt": "ChatGPT", "perplexity": "Perplexity", "claude": "Claude", "gemini": "Gemini", "grok": "Grok"}

    # Question generation — always dynamic, category-aware
    with st.spinner(f"Generating 10 buyer questions for '{category}'..."):
        brand_list_str = ", ".join(brands)
        gen_q = f"""You are building a brand visibility scanner. Your job is to generate exactly 10 questions that measure how well AI recommends brands in a category.

CATEGORY: "{category}"
BRANDS BEING TRACKED: {brand_list_str}

STRICT RULES — every question must follow ALL of these:

1. BRAND-INTENT REQUIRED: Every question must be asking for a brand recommendation, brand comparison across the full field, or which brand is best for something. The answer must naturally require naming one or more brands.

2. NO BRAND NAMES IN THE QUESTION: Do not mention any specific brand name in the question itself. The question must be open — any brand could win. Bad example: "Is Nike better than Adidas for running?" Good example: "What running shoe brand do podiatrists recommend most?"

3. NO HEAD-TO-HEAD COMPARISONS: Do not write questions that pit exactly two brands against each other. Those questions only produce two brand names and distort scoring for everyone else. Avoid "X vs Y" or "which is better, X or Y" formats entirely.

4. NO GENERIC EDUCATION QUESTIONS: Do not write questions about how to do something, what a feature means, or general category education. Bad: "How do I choose the right running shoe?" Good: "Which running shoe brand is best for beginners?"

5. COVER DIFFERENT BUYING CONTEXTS: Spread questions across different use cases, buyer types, price points, and decision contexts so different brands have a fair chance to appear. Examples of good dimensions: best overall, best for a specific use case, best for a specific person type, best value, best premium, most recommended by experts, most popular in a community, best for a specific need.

6. MATCH THE CATEGORY TYPE: Consumer product → shopper questions. B2B software → business buyer questions. Service → evaluation questions. Do not use software jargon for consumer categories.

Return ONLY the 10 questions as a numbered list. No intro, no explanation, nothing else."""

        if "gpt" in llm_map:
            raw = query_openai(gen_q, llm_map["gpt"]["args"][0])
        elif "claude" in llm_map:
            raw = query_claude(gen_q, claude_key)
        elif "gemini" in llm_map:
            raw = query_gemini(gen_q, gemini_key)
        else:
            raw = query_perplexity(gen_q, perp_key)

        generated_questions = [
            line.split(".", 1)[-1].strip().strip('"')
            for line in raw.strip().split("\n")
            if line.strip() and line.strip()[0].isdigit()
        ][:10]

        # Filter out questions that name specific brands or look like head-to-heads
        brand_names_lower = [b.lower() for b in brands]
        vs_pattern = re.compile(r'\bvs\.?\b|\bversus\b', re.IGNORECASE)

        def is_valid_question(q):
            q_lower = q.lower()
            # Reject if it names any of the tracked brands
            if any(b in q_lower for b in brand_names_lower):
                return False
            # Reject if it's a head-to-head
            if vs_pattern.search(q):
                return False
            # Reject if it's too short to be a real question
            if len(q.split()) < 5:
                return False
            return True

        generated_questions = [q for q in generated_questions if is_valid_question(q)]

        if len(generated_questions) < 5:
            st.error(f"Only {len(generated_questions)} valid questions generated — need at least 5. Try rephrasing your category (e.g. 'CRM Software for Small Business Teams' instead of 'CRM for Small Business') or check that your API key is working.")
            st.stop()

        # Merge: AI-generated first, then custom (validated + deduplicated)
        seen = set(q.lower() for q in generated_questions)

        valid_custom = []
        rejected_custom = []
        for q in custom_questions:
            if q.lower() in seen:
                continue  # duplicate, skip silently
            if not is_valid_question(q):
                # Figure out why
                q_lower = q.lower()
                if any(b in q_lower for b in brand_names_lower):
                    reason = "contains a brand name — questions must be open so any brand can win"
                elif vs_pattern.search(q):
                    reason = "head-to-head comparison — only 2 brands would appear, distorting scores"
                else:
                    reason = "no brand-seeking intent — question wouldn't naturally produce brand recommendations"
                rejected_custom.append((q, reason))
            else:
                valid_custom.append(q)
                seen.add(q.lower())

        questions = generated_questions + valid_custom

        if rejected_custom:
            for q, reason in rejected_custom:
                st.warning(f"**Custom question skipped:** \"{q}\"\n\n↳ {reason}")

        label = f"📋 {len(generated_questions)} AI-generated"
        if valid_custom:
            label += f" + {len(valid_custom)} custom = {len(questions)} total questions for '{category}'"
        else:
            label += f" questions for '{category}' — click to preview"

        with st.expander(label):
            if generated_questions:
                st.markdown(
                    "<div style='font-family: DM Mono; font-size: 0.62rem; color: #444; "
                    "letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.4rem;'>"
                    "AI-Generated</div>",
                    unsafe_allow_html=True
                )
                for q in generated_questions:
                    st.markdown(
                        f"<div style='font-family: DM Mono; font-size: 0.78rem; color: #666; "
                        f"padding: 0.35rem 0; border-bottom: 1px solid #111;'>→ {q}</div>",
                        unsafe_allow_html=True
                    )
            if valid_custom:
                st.markdown(
                    "<div style='font-family: DM Mono; font-size: 0.62rem; color: #7c3aed; "
                    "letter-spacing: 0.08em; text-transform: uppercase; margin: 0.75rem 0 0.4rem;'>"
                    "Your Questions</div>",
                    unsafe_allow_html=True
                )
                for q in valid_custom:
                    st.markdown(
                        f"<div style='font-family: DM Mono; font-size: 0.78rem; color: #7c3aed; "
                        f"padding: 0.35rem 0; border-bottom: 1px solid #1a1a2e;'>→ {q}</div>",
                        unsafe_allow_html=True
                    )

    st.markdown("---")
    active_label = " + ".join(llm_labels.get(k, k.title()) for k in llm_keys)
    st.markdown(f'<div class="section-label">Running scan — querying {len(llm_keys)} AI engine{"s" if len(llm_keys) > 1 else ""} ({active_label}) × {len(questions)} questions</div>', unsafe_allow_html=True)

    results = []
    progress = st.progress(0)
    status = st.empty()
    log_container = st.empty()
    log_html = ""

    for i, question in enumerate(questions):
        status.markdown(
            f'<div style="font-family: DM Mono; font-size: 0.75rem; color: #7c3aed;">'
            f'Querying question {i+1}/{len(questions)}...</div>',
            unsafe_allow_html=True
        )

        row = {"question": question}
        all_mentioned = set()
        llm_errors = []

        for llm, cfg in llm_map.items():
            if cfg.get("key_type") == "client":
                response = cfg["fn"](question, cfg["args"][0])
            else:
                response = cfg["fn"](question, cfg["args"][0])
            row[f"{llm}_response"] = response
            if response.startswith("ERROR:"):
                llm_errors.append((llm, response))
            else:
                m = count_mentions(response, brands)
                for b in brands:
                    if m[b] > 0:
                        all_mentioned.add(b)

        mentioned_str = ", ".join(f"<strong>{b}</strong>" for b in sorted(all_mentioned)) if all_mentioned else "<em style='color:#555'>none</em>"
        error_str = ""
        if llm_errors:
            error_str = " · " + " · ".join(
                f"<span style='color:#ef4444'>{llm_labels.get(llm, llm)} failed</span>"
                for llm, _ in llm_errors
            )
        log_html += f"""
        <div class="question-row">
            Q{i+1}: {question}<br>
            <span style='font-family: DM Mono; font-size: 0.7rem; color: #7c3aed;'>
                Brands mentioned: {mentioned_str}{error_str}
            </span>
        </div>"""
        log_container.markdown(log_html, unsafe_allow_html=True)

        results.append(row)
        progress.progress((i + 1) / len(questions))

    status.empty()
    progress.empty()

    # ── Build scorecard ───────────────────────────────────────────────────────
    df = build_scorecard(results, brands, llm_keys)

    st.markdown("---")
    st.markdown("""
    <div style='font-family: DM Mono; font-size: 0.68rem; letter-spacing: 0.15em;
                color: #555; text-transform: uppercase; margin-bottom: 1.5rem;'>
        GEO Scorecard — AI Visibility Results
    </div>
    """, unsafe_allow_html=True)

    # Top score cards
    top_n = min(len(df), 5)
    score_cols = st.columns(top_n)
    score_colors = ["#22c55e", "#86efac", "#f59e0b", "#f87171", "#ef4444"]

    for i, col in enumerate(score_cols):
        if i < len(df):
            row = df.iloc[i]
            color = score_colors[i] if row["Visibility Score"] > 0 else "#333"
            with col:
                st.markdown(f"""
                <div class="score-card">
                    <div class="score-number" style="color: {color};">{row['Visibility Score']}</div>
                    <div class="score-label">Visibility Score</div>
                    <div class="score-brand">{row['Brand']}</div>
                    <div style="font-family: DM Mono; font-size: 0.68rem; color: #555; margin-top: 0.4rem;">
                        {row['Questions Appeared In']}/{len(questions)} questions
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2, gap="large")

    with chart_col1:
        st.markdown('<div class="section-label">Overall Visibility Score</div>', unsafe_allow_html=True)
        st.plotly_chart(make_bar_chart(df), use_container_width=True, config={"displayModeBar": False})

    with chart_col2:
        st.markdown('<div class="section-label">Mentions by AI Engine</div>', unsafe_allow_html=True)
        st.plotly_chart(make_mention_comparison(df, llm_keys), use_container_width=True, config={"displayModeBar": False})

    # ── Full table ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Full Scorecard</div>', unsafe_allow_html=True)
    mention_cols = [f"{k.title()} Mentions" for k in llm_keys]
    display_cols = ["Brand", "Visibility Score", "Questions Appeared In", "Total Mentions"] + mention_cols + ["AI Recommendation Risk"]
    display_df = df[[c for c in display_cols if c in df.columns]].copy()
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=False,
        column_config={
            "Visibility Score": st.column_config.ProgressColumn(
                "Visibility Score", min_value=0, max_value=100, format="%d"
            )
        }
    )

    # ── Winner / loser callouts ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">Key Findings</div>', unsafe_allow_html=True)

    winner = df.iloc[0]
    loser = df[df["Questions Appeared In"] == 0]
    weak = df[(df["Questions Appeared In"] > 0) & (df["Questions Appeared In"] <= 3)]

    st.markdown(f"""
    <div class="winner-box">
        <div class="callout-tag" style="color: #22c55e;">🏆 AI-Dominant Brand</div>
        <div class="callout-text">
            <strong>{winner['Brand']}</strong> appeared in {winner['Questions Appeared In']}/{len(questions)} buyer questions
            with a visibility score of <strong>{winner['Visibility Score']}</strong>.
            When buyers ask AI about {category}, this brand is in the conversation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if len(loser) > 0:
        invisible = ", ".join(loser["Brand"].tolist())
        st.markdown(f"""
        <div class="loser-box">
            <div class="callout-tag" style="color: #ef4444;">🔴 AI-Invisible Brands</div>
            <div class="callout-text">
                <strong>{invisible}</strong> — appeared in <strong>0 out of {len(questions)}</strong> buyer questions
                across {len(llm_keys)} AI engine{"s" if len(llm_keys) > 1 else ""}. Zero AI search presence in this category.
                No amount of SEO fixes this.
            </div>
        </div>
        """, unsafe_allow_html=True)

    if len(weak) > 0:
        weak_brands = ", ".join(weak["Brand"].tolist())
        st.markdown(f"""
        <div style="background: #13100a; border: 1px solid #3a2a0a; border-left: 3px solid #f59e0b;
                    border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 0.75rem;">
            <div class="callout-tag" style="color: #f59e0b;">🟡 Weak AI Presence</div>
            <div class="callout-text">
                <strong>{weak_brands}</strong> — appeared in 3 or fewer questions.
                Present but easily displaced. Vulnerable to brands investing in GEO.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── GEO Playbook — always for your brand ─────────────────────────────────
    st.markdown("---")
    st.markdown(f'<div class="section-label">Your GEO Playbook — {your_brand.strip()}</div>', unsafe_allow_html=True)

    selected_brand = your_brand.strip()
    total_q = len(questions)

    # Get your brand's row — handle if brand didn't appear at all
    if selected_brand in df["Brand"].values:
        brand_row = df[df["Brand"] == selected_brand].iloc[0]
        score = brand_row["Visibility Score"]
        appeared = brand_row["Questions Appeared In"]
    else:
        score = 0
        appeared = 0

    missed_q = total_q - appeared
    gap_vs_winner = winner["Visibility Score"] - score

    # Score tier color
    if score == 0:
        tier_label = "🔴 Invisible"
        tier_color = "#ef4444"
    elif score <= 30:
        tier_label = f"🟡 Weak Presence"
        tier_color = "#f59e0b"
    elif score <= 60:
        tier_label = f"🟠 Emerging"
        tier_color = "#f97316"
    else:
        tier_label = f"🟢 Visible"
        tier_color = "#22c55e"

    st.markdown(f"""
    <div style="background: #0d0d18; border: 1px solid #1e1e2e; border-left: 3px solid {tier_color};
                border-radius: 8px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;">
        <div style="font-family: DM Mono; font-size: 0.68rem; color: {tier_color}; letter-spacing: 0.1em; margin-bottom: 0.5rem;">
            {selected_brand} · {tier_label} · Score {score}/100
        </div>
        <div style="font-family: Syne; font-size: 0.9rem; color: #aaa; line-height: 1.6;">
            Appeared in <strong style="color:#e8e8f0">{appeared}/{total_q}</strong> buyer questions
            across {len(llm_keys)} AI engine{"s" if len(llm_keys) > 1 else ""}.
            {f"<strong style='color:#ef4444'>Gap vs leader ({winner['Brand']}): {gap_vs_winner} points.</strong>" if gap_vs_winner > 0 else "<strong style='color:#22c55e'>You are the category leader.</strong>"}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Missed questions
    missed_questions = []
    for r in results:
        all_responses = " ".join(r.get(f"{llm}_response", "") for llm in llm_keys)
        if selected_brand.lower() not in all_responses.lower():
            missed_questions.append(r["question"])

    if missed_questions:
        with st.expander(f"📋 {len(missed_questions)} questions {selected_brand} didn't appear in — each one is a content brief"):
            for q in missed_questions:
                st.markdown(
                    f"<div style='font-family: DM Mono; font-size: 0.78rem; color: #666; "
                    f"padding: 0.4rem 0; border-bottom: 1px solid #111;'>→ {q}</div>",
                    unsafe_allow_html=True
                )

    # Build context — what LLMs said about your brand vs winner vs top competitors
    winner_responses_sample = []
    brand_responses_sample = []
    competitor_insights = []

    for r in results[:5]:
        for llm in llm_keys:
            resp = r.get(f"{llm}_response", "")
            if winner["Brand"].lower() in resp.lower() and winner["Brand"] != selected_brand:
                winner_responses_sample.append(f"Q: {r['question']}\nAI on {winner['Brand']}: {resp[:250]}")
                break
        for llm in llm_keys:
            resp = r.get(f"{llm}_response", "")
            if selected_brand.lower() in resp.lower():
                brand_responses_sample.append(f"Q: {r['question']}\nAI on {selected_brand}: {resp[:250]}")
                break

    # What competitors scored higher are doing — for competitive intel section
    higher_brands = df[df["Visibility Score"] > score].head(3)
    for _, hb in higher_brands.iterrows():
        hb_name = hb["Brand"]
        hb_responses = []
        for r in results[:3]:
            for llm in llm_keys:
                resp = r.get(f"{llm}_response", "")
                if hb_name.lower() in resp.lower():
                    hb_responses.append(f"Q: {r['question']}\nAI said: {resp[:200]}")
                    break
        if hb_responses:
            competitor_insights.append(f"{hb_name} (score {hb['Visibility Score']}):\n" + "\n".join(hb_responses[:2]))

    # ── AI Playbook prompt ────────────────────────────────────────────────────
    playbook_prompt = f"""You are a GEO (Generative Engine Optimization) strategist. Generate a specific, actionable optimization playbook for a brand based on real AI scan data.

SCAN DATA:
- Category: {category}
- Brand: {selected_brand}
- Visibility Score: {score}/100
- Appeared in: {appeared}/{total_q} buyer questions
- Missed {missed_q} questions
- Gap vs leader ({winner['Brand']}): {gap_vs_winner} points
- LLMs scanned: {", ".join(llm_labels.get(k, k) for k in llm_keys)}

QUESTIONS {selected_brand.upper()} MISSED:
{chr(10).join(f"- {q}" for q in missed_questions) if missed_questions else "None — appeared in all questions"}

HOW AI TALKED ABOUT {selected_brand.upper()} when it appeared:
{chr(10).join(brand_responses_sample) if brand_responses_sample else "Did not appear in any responses"}

HOW AI TALKED ABOUT THE LEADER ({winner['Brand'].upper()}):
{chr(10).join(winner_responses_sample) if winner_responses_sample else "Not captured"}

Generate exactly 4 optimization recommendations for {selected_brand}. Rules:
- Be specific to this brand, this category, and these actual missed questions
- Reference the real gaps you see in the data
- Do NOT give generic advice like "get more reviews" without connecting it to the actual scan findings
- Tone: direct, strategic, like a senior consultant who studied the data

Return ONLY a valid JSON array with exactly 4 objects, each with keys:
- "priority": one of "🚨 URGENT", "🔥 HIGH", "📌 MEDIUM", "🛡️ DEFEND"
- "title": action title, max 10 words
- "why": 2-3 sentences explaining this specific gap based on the scan data
- "how": 3-4 sentences of concrete steps specific to this category and situation
- "effort": time estimate e.g. "3-5 days", "2-3 weeks"
- "impact": one of "High", "Medium-High", "Medium", "Defensive", "Growth"

Raw JSON only. No markdown fences, no explanation."""

    with st.spinner(f"Analyzing scan data and generating playbook for {selected_brand}..."):
        if "gpt" in llm_map:
            raw_playbook = query_openai(playbook_prompt, llm_map["gpt"]["args"][0])
        elif "claude" in llm_map:
            raw_playbook = query_claude(playbook_prompt, claude_key)
        elif "gemini" in llm_map:
            raw_playbook = query_gemini(playbook_prompt, gemini_key)
        else:
            raw_playbook = query_perplexity(playbook_prompt, perp_key)

    try:
        clean = re.sub(r"```json|```", "", raw_playbook).strip()
        steps = json.loads(clean)
    except Exception:
        st.markdown(
            f"<div style='font-family: DM Mono; font-size: 0.78rem; color: #aaa; "
            f"background: #0a0a14; padding: 1rem; border-radius: 6px; line-height: 1.7;'>"
            f"{raw_playbook}</div>",
            unsafe_allow_html=True
        )
        steps = []

    for step in steps:
        impact = step.get("impact", "Medium")
        impact_color = "#22c55e" if impact == "High" else "#f59e0b" if "Medium" in impact else "#888"
        st.markdown(f"""
        <div style="background: #0a0a0f; border: 1px solid #1e1e2e; border-radius: 8px;
                    padding: 1.25rem 1.5rem; margin-bottom: 0.85rem;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.6rem;">
                <div style="font-family: DM Mono; font-size: 0.68rem; color: #7c3aed; letter-spacing: 0.08em;">
                    {step.get('priority', '')}
                </div>
                <div style="display: flex; gap: 0.5rem;">
                    <span style="font-family: DM Mono; font-size: 0.62rem; color: #444;
                                 background: #111; padding: 0.15rem 0.5rem; border-radius: 4px;">
                        ⏱ {step.get('effort', '')}
                    </span>
                    <span style="font-family: DM Mono; font-size: 0.62rem; color: {impact_color};
                                 background: #111; padding: 0.15rem 0.5rem; border-radius: 4px;">
                        ↑ {impact}
                    </span>
                </div>
            </div>
            <div style="font-family: Syne; font-size: 0.95rem; color: #e8e8f0; font-weight: 600; margin-bottom: 0.5rem;">
                {step.get('title', '')}
            </div>
            <div style="font-family: DM Mono; font-size: 0.75rem; color: #666; line-height: 1.6; margin-bottom: 0.6rem;">
                <strong style="color: #555;">Why this matters:</strong> {step.get('why', '')}
            </div>
            <div style="font-family: DM Mono; font-size: 0.75rem; color: #7c3aed; line-height: 1.6;">
                <strong>How:</strong> {step.get('how', '')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Competitive Intelligence ──────────────────────────────────────────────
    if competitor_insights and gap_vs_winner > 0:
        st.markdown("---")
        st.markdown(f'<div class="section-label">What higher-scoring competitors are doing differently</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-family: DM Mono; font-size: 0.72rem; color: #555; margin-bottom: 1.25rem;">'
            f'Based on how AI described them vs how it described {selected_brand} — patterns worth considering for your own strategy.'
            f'</div>',
            unsafe_allow_html=True
        )

        comp_intel_prompt = f"""You are a GEO strategist. Based on how AI talked about competing brands vs the brand being analyzed, identify specific patterns the brand could learn from and adapt for their own strategy.

BRAND BEING ANALYZED: {selected_brand} (Score: {score}/100)
CATEGORY: {category}

HOW AI TALKED ABOUT HIGHER-SCORING COMPETITORS:
{chr(10).join(competitor_insights)}

HOW AI TALKED ABOUT {selected_brand.upper()}:
{chr(10).join(brand_responses_sample) if brand_responses_sample else "Did not appear in responses"}

Identify exactly 3 specific patterns from how AI describes the higher-scoring brands that {selected_brand} could adapt. Be nuanced — don't say "copy them", say what strategic approach or content angle they're using that creates AI visibility, and how {selected_brand} could develop their own version of it given their positioning.

Return ONLY a valid JSON array with exactly 3 objects, each with keys:
- "competitor": which brand this pattern comes from
- "pattern": what the competitor is doing that creates AI visibility (1-2 sentences)
- "adaptation": how {selected_brand} could develop their own version of this, specific to their brand positioning (2-3 sentences)

Raw JSON only. No markdown, no explanation."""

        with st.spinner("Analyzing competitor patterns..."):
            if "gpt" in llm_map:
                raw_intel = query_openai(comp_intel_prompt, llm_map["gpt"]["args"][0])
            elif "claude" in llm_map:
                raw_intel = query_claude(comp_intel_prompt, claude_key)
            elif "gemini" in llm_map:
                raw_intel = query_gemini(comp_intel_prompt, gemini_key)
            else:
                raw_intel = query_perplexity(comp_intel_prompt, perp_key)

        try:
            clean_intel = re.sub(r"```json|```", "", raw_intel).strip()
            intel_items = json.loads(clean_intel)
        except Exception:
            intel_items = []

        for item in intel_items:
            st.markdown(f"""
            <div style="background: #0a0a0f; border: 1px solid #1e1e2e; border-left: 3px solid #06b6d4;
                        border-radius: 8px; padding: 1.25rem 1.5rem; margin-bottom: 0.85rem;">
                <div style="font-family: DM Mono; font-size: 0.65rem; color: #06b6d4;
                            letter-spacing: 0.1em; margin-bottom: 0.5rem;">
                    PATTERN FROM {item.get('competitor', '').upper()}
                </div>
                <div style="font-family: DM Mono; font-size: 0.75rem; color: #555;
                            line-height: 1.6; margin-bottom: 0.75rem;">
                    <strong style="color:#444;">What they're doing:</strong> {item.get('pattern', '')}
                </div>
                <div style="font-family: DM Mono; font-size: 0.75rem; color: #06b6d4; line-height: 1.6;">
                    <strong>Your version:</strong> {item.get('adaptation', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Raw responses expander ────────────────────────────────────────────────
    with st.expander("View raw AI responses"):
        for i, r in enumerate(results):
            st.markdown(f"**Q{i+1}: {r['question']}**")
            resp_cols = st.columns(len(llm_keys))
            for j, llm in enumerate(llm_keys):
                with resp_cols[j]:
                    st.markdown(f"*{llm_labels.get(llm, llm.title())}:*")
                    st.markdown(
                        f"<div style='font-size: 0.82rem; color: #aaa; background: #0a0a14; "
                        f"padding: 0.75rem; border-radius: 6px; font-family: DM Mono;'>"
                        f"{r.get(f'{llm}_response', '')[:400]}...</div>",
                        unsafe_allow_html=True
                    )
            st.markdown("---")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #1e1e2e;
                font-family: DM Mono; font-size: 0.68rem; color: #333; text-align: center;">
        Built by Astha Harlalka · asthaharlalka.com · 
        GEO Visibility Tracker · Data from {active_label}
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Onboarding guide ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin: 2rem 0 1rem 0; font-family: DM Mono; font-size: 0.68rem;
                color: #555; letter-spacing: 0.15em; text-transform: uppercase;">
        How to run your first scan
    </div>
    """, unsafe_allow_html=True)

    steps_html = [
        {
            "number": "01",
            "title": "Add at least one API key",
            "desc": "Open the sidebar on the left. Add keys for any LLM you have access to — the scan runs with whatever you provide. You need at least one to start. Keys stay for your session and are never stored.",
            "links": "Get keys → <a href='https://platform.openai.com' target='_blank' style='color:#7c3aed'>OpenAI</a> · <a href='https://perplexity.ai/settings/api' target='_blank' style='color:#7c3aed'>Perplexity</a> · <a href='https://console.anthropic.com' target='_blank' style='color:#7c3aed'>Anthropic</a> · <a href='https://aistudio.google.com' target='_blank' style='color:#7c3aed'>Gemini</a>",
            "status": "required"
        },
        {
            "number": "02",
            "title": "Type your category",
            "desc": "Enter any category you want to scan — B2B software, consumer products, FMCG, retail, services. The app generates 10 buyer-intent questions tailored to that category automatically.",
            "links": "Examples: <span style='color:#7c3aed'>Running Shoes</span> · <span style='color:#7c3aed'>Specialty Coffee</span> · <span style='color:#7c3aed'>CRM Software</span> · <span style='color:#7c3aed'>GTM Automation</span> · <span style='color:#7c3aed'>Project Management Tools</span>",
            "status": "required"
        },
        {
            "number": "03",
            "title": "Enter your brand and competitors",
            "desc": "Add your own brand in the 'Your brand' field — it'll appear first in all results. Then add competitor brands separated by commas. The more brands you add, the more interesting the comparison.",
            "links": "Example: <span style='color:#7c3aed'>Your brand → PiVector</span> · <span style='color:#7c3aed'>Competitors → Salesforce, HubSpot, Pipedrive, Zoho</span>",
            "status": "required"
        },
        {
            "number": "04",
            "title": "Add custom questions (optional)",
            "desc": "Have specific questions you want to test? Add them one per line. They run alongside the AI-generated set. Questions are validated — if one doesn't meet the brand-intent rules, it'll be flagged with a reason.",
            "links": "Rule: questions must be open-ended, no brand names in the question, no head-to-head comparisons",
            "status": "optional"
        },
        {
            "number": "05",
            "title": "Hit Run GEO Scan",
            "desc": "The scan fires all questions at every active LLM simultaneously. Takes 60–120 seconds depending on how many LLMs and questions you're running. Watch the questions fire in real time, then see the full scorecard.",
            "links": "Cost: approximately $0.02–0.05 per scan across all active LLMs",
            "status": "action"
        },
    ]

    cols = st.columns(len(steps_html), gap="small")
    status_colors = {"required": "#7c3aed", "optional": "#06b6d4", "action": "#22c55e"}
    status_labels = {"required": "Required", "optional": "Optional", "action": "Go"}

    for i, (col, step) in enumerate(zip(cols, steps_html)):
        color = status_colors[step["status"]]
        label = status_labels[step["status"]]
        with col:
            st.markdown(f"""
            <div style="background: #0d0d18; border: 1px solid #1e1e2e; border-top: 3px solid {color};
                        border-radius: 8px; padding: 1.5rem 1.25rem; height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <div style="font-family: DM Mono; font-size: 1.6rem; font-weight: 700;
                                color: #1e1e2e; line-height: 1;">
                        {step['number']}
                    </div>
                    <div style="font-family: DM Mono; font-size: 0.6rem; color: {color};
                                background: #0a0a0f; border: 1px solid {color}33;
                                padding: 0.15rem 0.5rem; border-radius: 4px; letter-spacing: 0.08em;">
                        {label}
                    </div>
                </div>
                <div style="font-family: Syne; font-size: 0.9rem; font-weight: 700;
                            color: #e8e8f0; margin-bottom: 0.75rem; line-height: 1.3;">
                    {step['title']}
                </div>
                <div style="font-family: DM Mono; font-size: 0.72rem; color: #555;
                            line-height: 1.7; margin-bottom: 1rem;">
                    {step['desc']}
                </div>
                <div style="font-family: DM Mono; font-size: 0.65rem; color: #444;
                            line-height: 1.6; border-top: 1px solid #1a1a2e; padding-top: 0.75rem;">
                    {step['links']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # What you get section
    st.markdown("""
    <div style="margin: 2.5rem 0 1rem 0; font-family: DM Mono; font-size: 0.68rem;
                color: #555; letter-spacing: 0.15em; text-transform: uppercase;">
        What you get after the scan
    </div>
    """, unsafe_allow_html=True)

    outputs = [
        ("📊", "Visibility Scorecard", "Each brand scored 0–100 based on how many questions they appeared in, total mentions, and sentiment across all active LLMs."),
        ("📈", "Per-LLM Breakdown", "See exactly which AI engines recommend which brands. ChatGPT vs Claude vs Gemini vs Perplexity — they don't always agree."),
        ("🔍", "Missed Questions", "Every question your brand didn't appear in is listed. Each one is a content brief — a gap you can close."),
        ("🗺️", "Optimization Playbook", "Tiered action plan based on your score. What to do, why it works, how to do it, effort estimate, and impact rating."),
    ]

    out_cols = st.columns(4, gap="small")
    for col, (icon, title, desc) in zip(out_cols, outputs):
        with col:
            st.markdown(f"""
            <div style="background: #0a0a0f; border: 1px solid #1e1e2e;
                        border-radius: 8px; padding: 1.25rem; text-align: center;">
                <div style="font-size: 1.5rem; margin-bottom: 0.6rem;">{icon}</div>
                <div style="font-family: Syne; font-size: 0.85rem; font-weight: 700;
                            color: #e8e8f0; margin-bottom: 0.5rem;">{title}</div>
                <div style="font-family: DM Mono; font-size: 0.68rem; color: #444;
                            line-height: 1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #1e1e2e;
                font-family: DM Mono; font-size: 0.68rem; color: #333; text-align: center;">
        Built by <a href="https://asthaharlalka.com" target="_blank"
        style="color: #7c3aed; text-decoration: none;">Astha Harlalka</a> ·
        GEO Visibility Tracker · Part of a 7-project AI marketing tools series
    </div>
    """, unsafe_allow_html=True)
