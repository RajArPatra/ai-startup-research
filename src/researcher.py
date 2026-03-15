#!/usr/bin/env python3
"""
AI Startup World Researcher
- Scrapes 11 sources using Playwright (headless Chromium)
- AI analysis: Gemini 2.0 Flash (primary) → Groq Llama 3.3 70B (fallback)
- Saves 8 topic reports as a single weekly JSON file
- Optionally emails summary via Resend API
"""

import os
import re
import json
import argparse
import time
import requests as http_requests
from datetime import datetime, timezone
from pathlib import Path

from groq import Groq
from playwright.sync_api import sync_playwright

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATE_STR = datetime.now().strftime("%Y-%m-%d")

# Primary: Gemini 2.0 Flash (1.5M tokens/day free — massive headroom)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"

# Fallback: Groq Llama 3.3 70B (100K tokens/day free)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODEL = "llama-3.1-8b-instant"

RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
EMAIL_TO = [
    "dhamakanetflix@gmail.com",
    # "rpatra1267@gmail.com",
    # "binayakmahesh012@gmail.com",
    # "santanusekhar2001@gmail.com",
]

# 8 report topics
REPORT_TOPICS = {
    "1_Funding_MarketMap": {
        "title": "Funding & Market Map",
        "questions": [
            "Which AI startups raised Series A/B in the last 6 months? What's the avg round size trending?",
            "Who are the most active AI-focused VCs right now and what thesis are they backing?",
            "Which segments are getting overcrowded vs. still whitespace?",
        ],
    },
    "2_Foundation_Models": {
        "title": "Foundation Model Layer",
        "questions": [
            "What's the competitive landscape beyond OpenAI/Anthropic/Google?",
            "How is open-source vs. closed-source playing out? What's the moat argument?",
            "What's happening with cost curves — are inference costs dropping fast enough?",
        ],
    },
    "3_Application_Layer": {
        "title": "Application Layer — AI-Native Startups",
        "questions": [
            "Which verticals have breakout AI-native companies? (Legal, healthcare, code, sales, finance, HR)",
            "What's the 'wrapper' vs. 'real product' distinction?",
            "What distribution strategies are working? (PLG, enterprise sales, API-first, embedded AI)",
        ],
    },
    "4_Infrastructure": {
        "title": "Infrastructure / Picks-and-Shovels",
        "questions": [
            "Who's winning in AI infra? (Vector DBs, evaluation, observability, orchestration)",
            "What's the data pipeline stack for AI companies?",
            "GPU/compute economics — who's building alternatives to NVIDIA dependency?",
        ],
    },
    "5_AI_Agents": {
        "title": "AI Agents & Autonomy",
        "questions": [
            "Which companies are building agentic systems that actually work in production?",
            "What's the real state of AI agents vs. the hype?",
            "How are enterprises actually adopting agents?",
        ],
    },
    "6_Business_Model_GTM": {
        "title": "Business Model & GTM",
        "questions": [
            "What pricing models are emerging? (Per-seat, per-outcome, usage-based, hybrid)",
            "How are AI startups handling the 'margin problem'?",
            "What does retention/churn look like for AI products vs. traditional SaaS?",
        ],
    },
    "7_Talent_Teams": {
        "title": "Talent & Team Composition",
        "questions": [
            "What does a founding team look like for a successful AI startup?",
            "Where is AI talent flowing — big labs, startups, or corporate AI teams?",
            "What roles are startups hiring for most aggressively?",
        ],
    },
    "8_Regulation_Risk": {
        "title": "Regulation & Risk",
        "questions": [
            "How are EU AI Act, US executive orders, and other regulation shaping startup strategy?",
            "What are the IP/copyright risks startups are navigating?",
            "How are startups handling safety, bias, and trust?",
        ],
    },
}

# Sources to scrape
SOURCES = [
    {"name": "TechCrunch", "urls": ["https://techcrunch.com/category/artificial-intelligence/", "https://techcrunch.com/tag/ai-funding/"], "category": "News"},
    {"name": "The Verge", "urls": ["https://www.theverge.com/ai-artificial-intelligence"], "category": "News"},
    {"name": "Hacker News", "urls": ["https://news.ycombinator.com/"], "category": "Community"},
    {"name": "Crunchbase News", "urls": ["https://news.crunchbase.com/ai/"], "category": "Funding"},
    {"name": "Bens Bites", "urls": ["https://bensbites.beehiiv.com/"], "category": "Newsletter"},
    {"name": "Import AI", "urls": ["https://jack-clark.net/"], "category": "Newsletter"},
    {"name": "The Batch", "urls": ["https://www.deeplearning.ai/the-batch/"], "category": "Newsletter"},
    {"name": "The Neuron", "urls": ["https://www.theneurondaily.com/"], "category": "Newsletter"},
    {"name": "Semafor Tech", "urls": ["https://www.semafor.com/vertical/tech"], "category": "News"},
    {"name": "Reddit ML", "urls": ["https://old.reddit.com/r/MachineLearning/hot/"], "category": "Community"},
    {"name": "Latent Space", "urls": ["https://www.latent.space/"], "category": "Podcast/Blog"},
]


# ──────────────────────────────────────────────
# SCRAPING
# ──────────────────────────────────────────────
def scrape_all_sources():
    """Use Playwright to scrape all sources. Returns dict of {source_name: text_content}."""
    results = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )

        for source in SOURCES:
            name = source["name"]
            print(f"  Scraping: {name}...")
            all_text = []
            for url in source["urls"]:
                try:
                    page = context.new_page()
                    page.goto(url, timeout=30000, wait_until="domcontentloaded")
                    page.wait_for_timeout(3000)
                    content = extract_page_content(page, name)
                    all_text.append(content)
                    print(f"    ✓ {url} — {len(content)} chars")
                    page.close()
                except Exception as e:
                    print(f"    ✗ {url} — {e}")

            combined = "\n\n".join(all_text)
            results[name] = combined

        browser.close()

    return results


def extract_page_content(page, source_name):
    """Extract meaningful text content from a page based on source type."""
    if source_name == "Hacker News":
        items = page.query_selector_all(".athing")
        lines = []
        for item in items[:30]:
            title_el = item.query_selector(".titleline > a")
            score_el = page.query_selector(f"#score_{item.get_attribute('id')}")
            if title_el:
                title = title_el.inner_text()
                href = title_el.get_attribute("href") or ""
                score = score_el.inner_text() if score_el else "0 points"
                lines.append(f"[{score}] {title} | {href}")
        return f"Source: Hacker News\nDate: {DATE_STR}\n\n" + "\n".join(lines)

    elif source_name == "Reddit ML":
        items = page.query_selector_all("#siteTable .thing")
        lines = []
        for item in items[:30]:
            title_el = item.query_selector("a.title")
            score_el = item.query_selector(".score.unvoted")
            if title_el:
                title = title_el.inner_text()
                score = score_el.inner_text() if score_el else "•"
                lines.append(f"[{score}] {title}")
        return f"Source: Reddit r/MachineLearning\nDate: {DATE_STR}\n\n" + "\n".join(lines)

    else:
        page.evaluate("""
            document.querySelectorAll('nav, footer, header, aside, .sidebar, .ad, .cookie-banner, script, style, noscript')
                .forEach(el => el.remove());
        """)
        elements = page.query_selector_all("h1, h2, h3, h4, p, li, article, .post-title, .entry-title, .article-title")
        lines = []
        seen = set()
        for el in elements:
            text = el.inner_text().strip()
            if len(text) > 15 and text not in seen:
                seen.add(text)
                lines.append(text)
        return f"Source: {source_name}\nDate: {DATE_STR}\n\n" + "\n".join(lines[:200])


# ──────────────────────────────────────────────
# AI ANALYSIS — Gemini (primary) → Groq (fallback)
# ──────────────────────────────────────────────
def build_prompt(scraped_data, topic_info):
    """Build the analysis prompt from scraped data and topic info."""
    all_content = ""
    for source_name, text in scraped_data.items():
        all_content += f"\n\n===== {source_name} =====\n{text}"

    # Gemini has huge context, but keep lean for Groq fallback compatibility
    if len(all_content) > 15000:
        all_content = all_content[:15000] + "\n\n[... truncated ...]"

    questions_text = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(topic_info["questions"]))

    return f"""You are a senior AI industry research analyst writing an intelligence briefing.

TODAY'S DATE: {DATE_STR}

TOPIC: {topic_info['title']}

KEY RESEARCH QUESTIONS:
{questions_text}

Below is raw scraped content from 11 AI news sources collected today. Analyze this content and write a detailed intelligence report addressing the research questions above.

SCRAPED DATA:
{all_content}

INSTRUCTIONS:
- Write a structured report with clear sections addressing each research question
- Synthesize across sources — don't just list what each source said
- Identify trends, patterns, and notable signals
- Call out specific companies, deals, and numbers when available
- Note gaps — what important questions the data doesn't answer
- Be analytical and opinionated, not just descriptive
- If the scraped data is thin on a topic, say so honestly and provide your best analysis based on what's available
- Use markdown-style formatting: ## for sections, **bold** for emphasis, - for bullets
- Aim for 600-900 words total

Write the report now:"""


def analyze_with_gemini(prompt):
    """Call Gemini 2.0 Flash API. Returns (text, model_name) or raises on failure."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1500,
        },
    }

    for attempt in range(3):
        resp = http_requests.post(url, json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return text, f"Gemini 2.0 Flash"
        elif resp.status_code == 429:
            wait = 15 * (attempt + 1)
            print(f"    Gemini rate limited — waiting {wait}s (attempt {attempt+1}/3)...")
            time.sleep(wait)
        else:
            raise Exception(f"Gemini API error {resp.status_code}: {resp.text[:200]}")

    raise Exception("Gemini rate limited after 3 retries")


def analyze_with_groq(prompt):
    """Call Groq API. Returns (text, model_name) or raises on failure."""
    client = Groq(api_key=GROQ_API_KEY)

    for model in [GROQ_MODEL, GROQ_FALLBACK_MODEL]:
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500,
                    temperature=0.3,
                )
                return response.choices[0].message.content, f"Groq {model}"
            except Exception as e:
                err = str(e).lower()
                if "rate_limit" in err or "429" in err:
                    if "tokens per day" in err:
                        print(f"    Groq daily limit on {model} — trying next...")
                        break
                    wait = 30 * (attempt + 1)
                    print(f"    Groq rate limited — waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    raise Exception("All Groq models rate limited")


def analyze_topic(scraped_data, topic_info):
    """Analyze one topic: try Gemini first, fallback to Groq."""
    prompt = build_prompt(scraped_data, topic_info)

    # 1. Try Gemini (1.5M tokens/day — should always work)
    if GEMINI_API_KEY:
        try:
            text, model = analyze_with_gemini(prompt)
            print(f"    [{model}] ✓")
            return text, model
        except Exception as e:
            print(f"    Gemini failed: {e} — falling back to Groq...")

    # 2. Fallback to Groq (Llama 3.3 70B → Llama 3.1 8B)
    if GROQ_API_KEY:
        try:
            text, model = analyze_with_groq(prompt)
            print(f"    [{model}] ✓")
            return text, model
        except Exception as e:
            print(f"    Groq failed: {e}")

    return "Analysis unavailable — all AI providers failed. Will retry next run.", "none"


def run_all_analyses(scraped_data):
    """Run AI analysis for all 8 topics with Gemini → Groq fallback chain."""
    analyses = {}
    models_used = set()
    for topic_id, topic_info in REPORT_TOPICS.items():
        print(f"  Analyzing: {topic_info['title']}...")
        analysis, model = analyze_topic(scraped_data, topic_info)
        analyses[topic_id] = analysis
        models_used.add(model)
        print(f"    Done — {len(analysis)} chars")
        time.sleep(3)

    print(f"  Models used: {', '.join(models_used)}")
    return analyses


# ──────────────────────────────────────────────
# JSON OUTPUT
# ──────────────────────────────────────────────
def save_weekly_json(scraped_data, analyses, output_dir):
    """Write the weekly report as a single JSON file. Returns the file path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data = {
        "date": DATE_STR,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": GROQ_MODEL,
        "sources": [s["name"] for s in SOURCES],
        "reports": {},
        "scrape_stats": {
            "total_chars": sum(len(v) for v in scraped_data.values()),
            "source_count": len(scraped_data),
            "sources": {
                name: {"chars": len(text), "status": "ok" if len(text) > 100 else "thin"}
                for name, text in scraped_data.items()
            },
        },
    }

    for topic_id, topic_info in REPORT_TOPICS.items():
        data["reports"][topic_id] = {
            "title": topic_info["title"],
            "questions": topic_info["questions"],
            "analysis": analyses.get(topic_id, ""),
        }

    json_file = output_path / f"{DATE_STR}.json"
    json_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved: {json_file} ({json_file.stat().st_size / 1024:.1f} KB)")
    return json_file


def update_manifest(output_dir):
    """Update docs/data/index.json manifest with the new week."""
    manifest_path = Path(output_dir) / "index.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"weeks": []}

    # Don't duplicate
    existing_dates = {w["date"] for w in manifest["weeks"]}
    if DATE_STR not in existing_dates:
        manifest["weeks"].append({
            "date": DATE_STR,
            "file": f"{DATE_STR}.json",
            "topics": 8,
        })
        # Sort newest first
        manifest["weeks"].sort(key=lambda w: w["date"], reverse=True)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Manifest updated: {len(manifest['weeks'])} weeks total")


# ──────────────────────────────────────────────
# EMAIL (Resend — optional)
# ──────────────────────────────────────────────
def send_email(site_url=""):
    """Send a summary email via Resend with link to the website."""
    if not RESEND_API_KEY:
        print("  No RESEND_API_KEY — skipping email")
        return

    import resend
    resend.api_key = RESEND_API_KEY

    body = f"""AI Startup Research — Weekly Intelligence Report
Date: {DATE_STR}

8 topic reports have been generated and published:
1. Funding & Market Map
2. Foundation Model Layer
3. Application Layer — AI-Native Startups
4. Infrastructure / Picks-and-Shovels
5. AI Agents & Autonomy
6. Business Model & GTM
7. Talent & Team Composition
8. Regulation & Risk

View full reports: {site_url}#{DATE_STR}

Sources: TechCrunch, The Verge, Hacker News, Crunchbase News,
Ben's Bites, Import AI, The Batch, The Neuron, Semafor Tech,
Reddit r/MachineLearning, Latent Space

---
Auto-generated by AI Startup Researcher"""

    try:
        r = resend.Emails.send({
            "from": "AI Startup Researcher <onboarding@resend.dev>",
            "to": EMAIL_TO,
            "subject": f"AI Startup Intelligence — {DATE_STR}",
            "text": body,
        })
        print(f"  Email sent! ID: {r['id']}")
    except Exception as e:
        print(f"  Email failed: {e}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AI Startup Researcher")
    parser.add_argument("--output-dir", default="docs/data", help="Output directory for JSON files")
    parser.add_argument("--site-url", default="", help="Website URL for email links")
    args = parser.parse_args()

    print("=" * 55)
    print(f"  AI STARTUP RESEARCHER — {DATE_STR}")
    print("=" * 55)

    # Step 1: Scrape
    print("\n[1/4] Scraping 11 sources...")
    scraped = scrape_all_sources()
    total = sum(len(v) for v in scraped.values())
    print(f"  Total: {total:,} chars from {len(scraped)} sources")

    # Step 2: AI Analysis
    print("\n[2/4] AI analysis (8 topics)...")
    analyses = run_all_analyses(scraped)

    # Step 3: Save JSON
    print("\n[3/4] Saving JSON...")
    save_weekly_json(scraped, analyses, args.output_dir)
    update_manifest(args.output_dir)

    # Step 4: Email
    print("\n[4/4] Email...")
    send_email(args.site_url)

    print(f"\n{'=' * 55}")
    print("  DONE!")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
