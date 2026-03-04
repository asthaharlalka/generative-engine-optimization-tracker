---
title: GEO Visibility Tracker
emoji: 🔍
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# GEO Visibility Tracker

**Find out which brands AI recommends when your buyers ask category questions — before your competitors do.**

Built by [Astha Harlalka](https://asthaharlalka.com) · [Live Demo →](https://geo-tracker.asthaharlalka.com)

---

## What is GEO?

Generative Engine Optimization (GEO) is the discipline of making your brand visible inside AI-generated answers — not just search results.

94% of B2B buyers now use AI tools like ChatGPT, Perplexity, Claude, and Gemini to research vendors before making a purchase decision. If your brand isn't appearing in those answers, you're invisible to buyers who have already moved past Google.

This tool measures exactly that: which brands AI recommends when real buyers ask real questions in your category.

---

## What It Does

The GEO Visibility Tracker fires a set of buyer-intent questions at up to 5 AI engines simultaneously, counts how often each brand appears in the responses, and produces a visibility scorecard with an actionable optimization playbook — specific to your brand.

**Works for any category** — B2B software, consumer products, FMCG, services, retail. The question generation adapts to the type of category you enter.

---

## Features

- **Any category** — type anything: "Running Shoes", "GTM Automation Software", "Specialty Coffee", "Skincare for Oily Skin"
- **Up to 5 LLMs simultaneously** — ChatGPT, Perplexity, Claude, Gemini, Grok. Run with any combination you have keys for
- **Smart question generation** — 10 brand-intent questions generated and validated for your category. Questions with brand names, head-to-heads, or no brand intent are automatically rejected. Gemini uses JSON output mode for more reliable parsing
- **Your brand vs competitors** — enter your brand in Step 2 (required). It anchors the entire analysis
- **Custom questions** — add your own on top of the AI-generated set. Invalid questions are flagged with the specific reason they were rejected
- **Visibility scorecard** — each brand scored 0–100 based on breadth, frequency, and sentiment
- **Per-LLM breakdown** — see which AI engines are recommending which brands
- **GEO Optimization Playbook** — after every scan, get a 4-card AI-generated action plan for your brand, grounded in the actual scan data: missed questions, how AI described you vs the leader, specific steps to close the gap
- **Competitive intelligence** — separate section showing what higher-scoring brands are doing that yours isn't, framed as adaptations not copies
- **Stop scan** — cancel mid-scan without reloading the page. Partial results are shown
- **Session-persistent API keys** — keys stay for the duration of your tab session. Never stored, cleared on tab close

---

## How Scoring Works

Each brand is scored 0–100:

| Component | Weight | What it measures |
|---|---|---|
| Breadth | 60% | How many of the total questions the brand appeared in |
| Frequency | 25% | Total mention count across all LLMs (capped to prevent inflation) |
| Sentiment | 15% | Positive sentiment detected in the 250-character window around each brand mention (runs on first LLM response per question) |

**Risk tiers:**
- 🟢 Visible — appeared in 4+ questions
- 🟡 Weak — appeared in 1–3 questions
- 🔴 Invisible — appeared in 0 questions

---

## Question Quality Rules

Every question — AI-generated or custom — must pass all of these:

1. **Brand-intent required** — the answer must naturally require naming one or more brands
2. **No brand names in the question** — keeps the field open so any brand can win
3. **No head-to-head comparisons** — "X vs Y" questions only produce two brand names and distort scores
4. **No generic education questions** — "how do I choose a CRM" produces no brand recommendations
5. **Context-appropriate** — consumer questions for consumer categories, B2B questions for software

---

## Tech Stack

- [Streamlit](https://streamlit.io) — UI and app framework
- [OpenAI API](https://platform.openai.com) — GPT-4o-mini
- [Perplexity API](https://perplexity.ai) — sonar model
- [Anthropic API](https://anthropic.com) — Claude 3 Haiku
- [Google Gemini API](https://aistudio.google.com) — Gemini 2.5 Flash
- [xAI Grok API](https://console.x.ai) — Grok 4.1 Fast Reasoning
- [Plotly](https://plotly.com) — interactive charts
- [Pandas](https://pandas.pydata.org) — data processing

---

## Setup

### Prerequisites

- Python 3.9+
- API keys for at least one supported LLM

### Local Installation

```bash
git clone https://github.com/asthaharlalka/generative-engine-optimization-tracker
cd generative-engine-optimization-tracker
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

App opens at `http://localhost:8501`.

### API Keys

- **OpenAI** → [platform.openai.com](https://platform.openai.com)
- **Perplexity** → [perplexity.ai/settings/api](https://perplexity.ai/settings/api)
- **Anthropic** → [console.anthropic.com](https://console.anthropic.com)
- **Gemini** → [aistudio.google.com](https://aistudio.google.com)
- **Grok** → [console.x.ai](https://console.x.ai)

Keys are entered in the sidebar each session and are never stored or logged.

---

## Deployment

### Hugging Face Spaces (recommended)

1. Go to [huggingface.co](https://huggingface.co) → New Space
2. Select **Streamlit** as the SDK, set visibility to **Public**
3. Push this repo to the Space's git remote
4. HF auto-builds on push — live in ~2 minutes

### Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set main file as `app.py` and deploy
4. Add custom domain in Streamlit app settings
5. Add a CNAME record on your domain pointing to your Streamlit app URL

---

## Cost Per Scan

A typical 10-question scan across all 5 LLMs costs approximately $0.02–0.05. The playbook and competitive intelligence sections each make one additional API call (~$0.01 each). GPT-4o-mini and Claude Haiku are most cost-efficient. Note: Grok 4.1 Fast Reasoning is a reasoning model and may cost more per call than standard models.

---

## What GEO Optimization Actually Looks Like

After your scan, the app generates a playbook grounded in your actual results — which questions you missed, how AI described your competitors vs you, and what to do about it.

**LLMs recommend brands that:**
- Have structured content directly answering buyer questions (FAQ pages, comparison pages, use-case pages)
- Are cited across authoritative third-party sources (G2, Reddit, Capterra, industry blogs, press)
- Use language that matches how buyers actually phrase questions
- Appear in training data through reviews, case studies, and analyst mentions

**The biggest levers by score tier:**
- Score 0 → Get on G2, build comparison pages, answer Reddit questions, get press mentions
- Score 1–30 → Deepen use-case content, build "vs competitor" pages, increase review recency
- Score 31–60 → Fill specific question gaps, get cited by LLM-visible sources, publish original research
- Score 60+ → Defend position, expand to adjacent categories, use your score as a marketing asset

---

## License

MIT — use it, fork it, build on it.
