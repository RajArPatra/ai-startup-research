#!/usr/bin/env python3
"""
AI Research Paper Scraper & Analyzer
- Scrapes 8 sources: arXiv, HuggingFace, Import AI, TLDR AI, The Batch, PubMed, BioRxiv, Reddit ML
- AI analysis: Gemini 2.5 Flash (primary) → Groq Llama 3.3 70B (fallback)
- Saves 7 topic reports as a single weekly JSON file
"""

import os
import re
import json
import argparse
import time
import xml.etree.ElementTree as ET
import requests as http_requests
import feedparser
from datetime import datetime, timezone, timedelta
from pathlib import Path

from groq import Groq
from playwright.sync_api import sync_playwright

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATE_STR = datetime.now().strftime("%Y-%m-%d")
DATE_YESTERDAY = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
DATE_WEEK_AGO = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODEL = "llama-3.1-8b-instant"

USER_AGENT = "AI-Research-Bot/1.0 (academic research project)"

# 7 research topics with per-topic source filtering
RESEARCH_TOPICS = {
    "1_Trending_Papers": {
        "title": "Trending Papers & Preprints",
        "questions": [
            "What are the most discussed/upvoted AI papers this week across arXiv, HuggingFace, and Reddit?",
            "Which papers are getting traction in industry vs. academia? What's driving the attention?",
            "Are there any surprising papers from non-obvious institutions or authors?",
        ],
        "sources": ["arXiv", "HuggingFace Daily Papers", "Reddit ML"],
    },
    "2_Breakthrough_Models": {
        "title": "Breakthrough Models & Architectures",
        "questions": [
            "What new model architectures, training techniques, or scaling approaches were published this week?",
            "How do new models compare on key benchmarks (MMLU, HumanEval, MT-Bench, etc.)?",
            "What's the state of multimodal models, reasoning models, and long-context architectures?",
        ],
        "sources": ["arXiv", "HuggingFace Daily Papers", "The Batch", "TLDR AI", "Reddit ML"],
    },
    "3_OpenSource_Ecosystem": {
        "title": "Open-Source AI Developments",
        "questions": [
            "What significant open-source model releases, tools, or datasets launched this week?",
            "How are open-weight models (Llama, Mistral, Qwen, Gemma) progressing vs. closed models?",
            "Which open-source libraries, frameworks, or tools are gaining adoption?",
        ],
        "sources": ["HuggingFace Daily Papers", "Reddit ML", "Import AI", "TLDR AI", "arXiv"],
    },
    "4_AI_Safety_Alignment": {
        "title": "AI Safety, Alignment & Governance",
        "questions": [
            "What new safety research, red-teaming results, or alignment papers were published?",
            "How are labs and governments addressing AI risk — new policies, standards, or frameworks?",
            "What's the latest on interpretability, RLHF improvements, and evaluation of dangerous capabilities?",
        ],
        "sources": ["arXiv", "Import AI", "The Batch", "TLDR AI", "Reddit ML"],
    },
    "5_Applied_AI_Healthcare": {
        "title": "AI in Healthcare, Biology & Science",
        "questions": [
            "What are the most promising AI applications in drug discovery, diagnostics, or clinical trials?",
            "Which biomedical AI papers show real clinical impact or novel biological insights?",
            "How is AI being used in protein folding, genomics, medical imaging, or scientific simulation?",
        ],
        "sources": ["PubMed", "BioRxiv", "arXiv", "The Batch"],
    },
    "6_Benchmarks_Evaluation": {
        "title": "Benchmarks, Evaluations & Datasets",
        "questions": [
            "What new benchmarks or evaluation frameworks were proposed this week?",
            "How are existing leaderboards shifting — which models are climbing/falling?",
            "What new datasets were released, and what gaps do they address?",
        ],
        "sources": ["arXiv", "HuggingFace Daily Papers", "Reddit ML"],
    },
    "7_Industry_Applications": {
        "title": "Industry & Applied AI Trends",
        "questions": [
            "What AI product launches, enterprise deployments, or real-world applications made news?",
            "How are companies integrating AI agents, copilots, or automation into workflows?",
            "What emerging AI use cases are gaining traction beyond chatbots and code generation?",
        ],
        "sources": ["The Batch", "TLDR AI", "Import AI", "Reddit ML", "HuggingFace Daily Papers"],
    },
}

SOURCE_NAMES = [
    "arXiv", "HuggingFace Daily Papers", "Import AI", "TLDR AI",
    "The Batch", "PubMed", "BioRxiv", "Reddit ML",
]


# ──────────────────────────────────────────────
# SCRAPING — API-based sources
# ──────────────────────────────────────────────
def scrape_arxiv():
    """Fetch latest AI papers from arXiv API."""
    url = (
        "https://export.arxiv.org/api/query?"
        "search_query=cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.CL"
        "&sortBy=submittedDate&sortOrder=descending&max_results=50"
    )
    resp = http_requests.get(url, timeout=30)
    if resp.status_code != 200:
        return f"Source: arXiv\nDate: {DATE_STR}\n\nFailed to fetch: HTTP {resp.status_code}"

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    entries = root.findall("atom:entry", ns)

    lines = []
    for entry in entries[:50]:
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")[:500]
        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)][:5]
        published = entry.find("atom:published", ns).text[:10]
        arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
        lines.append(f"[{arxiv_id}] {title}\n  Authors: {', '.join(authors)}\n  Published: {published}\n  Abstract: {abstract}")

    return f"Source: arXiv (cs.AI, cs.LG, cs.CL)\nDate: {DATE_STR}\nPapers: {len(entries)}\n\n" + "\n\n".join(lines)


def scrape_huggingface():
    """Fetch trending daily papers from HuggingFace."""
    resp = http_requests.get("https://huggingface.co/api/daily_papers", timeout=30)
    if resp.status_code != 200:
        return f"Source: HuggingFace Daily Papers\nDate: {DATE_STR}\n\nFailed: HTTP {resp.status_code}"

    papers = resp.json()
    # Sort by upvotes, take top 30
    papers.sort(key=lambda p: p.get("paper", {}).get("upvotes", 0), reverse=True)

    lines = []
    for p in papers[:30]:
        paper = p.get("paper", {})
        title = paper.get("title", "Untitled")
        summary = paper.get("summary", "")[:500]
        upvotes = paper.get("upvotes", 0)
        authors = [a.get("name", "") for a in paper.get("authors", [])][:5]
        pub_date = paper.get("publishedAt", "")[:10]
        lines.append(f"[{upvotes} upvotes] {title}\n  Authors: {', '.join(authors)}\n  Published: {pub_date}\n  Summary: {summary}")

    return f"Source: HuggingFace Daily Papers\nDate: {DATE_STR}\nPapers: {len(papers[:30])}\n\n" + "\n\n".join(lines)


def scrape_import_ai():
    """Fetch Import AI newsletter via Substack RSS."""
    feed = feedparser.parse("https://importai.substack.com/feed")
    if not feed.entries:
        return f"Source: Import AI\nDate: {DATE_STR}\n\nNo entries found"

    lines = []
    for entry in feed.entries[:5]:
        title = entry.get("title", "Untitled")
        published = entry.get("published", "")
        # Strip HTML tags from content
        content = entry.get("summary", "") or entry.get("description", "")
        content = re.sub(r"<[^>]+>", " ", content)
        content = re.sub(r"\s+", " ", content).strip()[:2000]
        lines.append(f"Title: {title}\nPublished: {published}\nContent: {content}")

    return f"Source: Import AI (Substack)\nDate: {DATE_STR}\nArticles: {len(lines)}\n\n" + "\n\n---\n\n".join(lines)


def scrape_pubmed():
    """Fetch latest AI/ML papers from PubMed E-utilities."""
    # Step 1: Search for IDs
    search_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        "db=pubmed&term=artificial+intelligence+OR+machine+learning+OR+deep+learning"
        "&retmax=30&sort=date&retmode=json"
    )
    resp = http_requests.get(search_url, timeout=30)
    if resp.status_code != 200:
        return f"Source: PubMed\nDate: {DATE_STR}\n\nSearch failed: HTTP {resp.status_code}"

    ids = resp.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return f"Source: PubMed\nDate: {DATE_STR}\n\nNo results found"

    time.sleep(0.5)  # Rate limit: 3 req/sec without API key

    # Step 2: Fetch abstracts
    fetch_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
        f"db=pubmed&id={','.join(ids[:30])}&retmode=xml"
    )
    resp = http_requests.get(fetch_url, timeout=30)
    if resp.status_code != 200:
        return f"Source: PubMed\nDate: {DATE_STR}\n\nFetch failed: HTTP {resp.status_code}"

    root = ET.fromstring(resp.text)
    lines = []
    for article in root.findall(".//PubmedArticle"):
        title_el = article.find(".//ArticleTitle")
        abstract_el = article.find(".//AbstractText")
        authors = article.findall(".//Author")
        pmid_el = article.find(".//PMID")

        title = title_el.text if title_el is not None and title_el.text else "Untitled"
        abstract = abstract_el.text[:500] if abstract_el is not None and abstract_el.text else "No abstract"
        author_names = []
        for a in authors[:5]:
            last = a.find("LastName")
            first = a.find("ForeName")
            if last is not None and last.text:
                name = last.text + (f" {first.text[0]}" if first is not None and first.text else "")
                author_names.append(name)
        pmid = pmid_el.text if pmid_el is not None else ""

        lines.append(f"[PMID:{pmid}] {title}\n  Authors: {', '.join(author_names)}\n  Abstract: {abstract}")

    return f"Source: PubMed\nDate: {DATE_STR}\nPapers: {len(lines)}\n\n" + "\n\n".join(lines)


def scrape_biorxiv():
    """Fetch latest preprints from BioRxiv API."""
    url = f"https://api.biorxiv.org/details/biorxiv/{DATE_WEEK_AGO}/{DATE_STR}/0/30"
    resp = http_requests.get(url, timeout=30)
    if resp.status_code != 200:
        return f"Source: BioRxiv\nDate: {DATE_STR}\n\nFailed: HTTP {resp.status_code}"

    data = resp.json()
    collection = data.get("collection", [])

    # Filter for AI/ML-relevant papers by keyword matching
    ai_keywords = {"machine learning", "deep learning", "neural network", "artificial intelligence",
                   "transformer", "language model", "computer vision", "reinforcement learning",
                   "generative", "diffusion", "attention mechanism", "classification", "prediction model"}

    lines = []
    for paper in collection:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        combined = (title + " " + abstract).lower()

        # Include if any AI keyword matches
        if any(kw in combined for kw in ai_keywords):
            authors = paper.get("authors", "")[:200]
            doi = paper.get("doi", "")
            category = paper.get("category", "")
            lines.append(f"[{category}] {title}\n  Authors: {authors}\n  DOI: {doi}\n  Abstract: {abstract[:500]}")

    if not lines:
        # If no AI papers found, include first 10 papers anyway
        for paper in collection[:10]:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")[:500]
            authors = paper.get("authors", "")[:200]
            lines.append(f"{title}\n  Authors: {authors}\n  Abstract: {abstract}")

    return f"Source: BioRxiv\nDate: {DATE_STR}\nPapers: {len(lines)}\n\n" + "\n\n".join(lines)


def scrape_reddit_ml():
    """Fetch hot posts from r/MachineLearning via JSON API."""
    url = "https://www.reddit.com/r/MachineLearning/hot.json?limit=30"
    headers = {"User-Agent": USER_AGENT}
    resp = http_requests.get(url, headers=headers, timeout=30)
    if resp.status_code != 200:
        return f"Source: Reddit ML\nDate: {DATE_STR}\n\nFailed: HTTP {resp.status_code}"

    posts = resp.json().get("data", {}).get("children", [])
    lines = []
    for post in posts:
        d = post.get("data", {})
        title = d.get("title", "")
        score = d.get("score", 0)
        comments = d.get("num_comments", 0)
        url_link = d.get("url", "")
        flair = d.get("link_flair_text", "")
        selftext = (d.get("selftext", "") or "")[:300]

        flair_str = f"[{flair}] " if flair else ""
        lines.append(f"{flair_str}[{score} pts, {comments} comments] {title}\n  URL: {url_link}")
        if selftext:
            lines[-1] += f"\n  Text: {selftext}"

    return f"Source: Reddit r/MachineLearning\nDate: {DATE_STR}\nPosts: {len(lines)}\n\n" + "\n\n".join(lines)


# ──────────────────────────────────────────────
# SCRAPING — Playwright-based sources
# ──────────────────────────────────────────────
def scrape_playwright_sources():
    """Scrape TLDR AI and The Batch using Playwright. Returns dict."""
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

        # TLDR AI — last 2 days
        print("  Scraping: TLDR AI...")
        tldr_text = []
        for date in [DATE_STR, DATE_YESTERDAY]:
            try:
                page = context.new_page()
                page.goto(f"https://tldr.tech/ai/{date}", timeout=30000, wait_until="networkidle")
                page.wait_for_timeout(3000)
                text = page.inner_text("body")
                if len(text) > 500:  # Not a 404/empty page
                    tldr_text.append(f"--- {date} ---\n{text[:5000]}")
                    print(f"    ✓ tldr.tech/ai/{date} — {len(text)} chars")
                else:
                    print(f"    ✗ tldr.tech/ai/{date} — too short ({len(text)} chars)")
                page.close()
            except Exception as e:
                print(f"    ✗ tldr.tech/ai/{date} — {e}")

        results["TLDR AI"] = f"Source: TLDR AI Newsletter\nDate: {DATE_STR}\n\n" + "\n\n".join(tldr_text) if tldr_text else f"Source: TLDR AI\nDate: {DATE_STR}\n\nNo content available"

        # The Batch (deeplearning.ai)
        print("  Scraping: The Batch...")
        try:
            page = context.new_page()
            page.goto("https://www.deeplearning.ai/the-batch/", timeout=30000, wait_until="domcontentloaded")
            page.wait_for_timeout(3000)
            page.evaluate("""
                document.querySelectorAll('nav, footer, header, aside, .sidebar, .ad, .cookie-banner, script, style, noscript')
                    .forEach(el => el.remove());
            """)
            elements = page.query_selector_all("h1, h2, h3, h4, p, li, article, .post-title, .entry-title")
            lines = []
            seen = set()
            for el in elements:
                text = el.inner_text().strip()
                if len(text) > 15 and text not in seen:
                    seen.add(text)
                    lines.append(text)
            content = "\n".join(lines[:200])
            results["The Batch"] = f"Source: The Batch (deeplearning.ai)\nDate: {DATE_STR}\n\n{content}"
            print(f"    ✓ deeplearning.ai/the-batch — {len(content)} chars")
            page.close()
        except Exception as e:
            results["The Batch"] = f"Source: The Batch\nDate: {DATE_STR}\n\nFailed: {e}"
            print(f"    ✗ The Batch — {e}")

        browser.close()

    return results


# ──────────────────────────────────────────────
# MAIN SCRAPER
# ──────────────────────────────────────────────
def scrape_all_sources():
    """Scrape all 8 sources. Returns dict of {source_name: text_content}."""
    results = {}

    # API-based sources (no browser needed)
    api_scrapers = [
        ("arXiv", scrape_arxiv),
        ("HuggingFace Daily Papers", scrape_huggingface),
        ("Import AI", scrape_import_ai),
        ("PubMed", scrape_pubmed),
        ("BioRxiv", scrape_biorxiv),
        ("Reddit ML", scrape_reddit_ml),
    ]

    for name, scraper in api_scrapers:
        print(f"  Scraping: {name}...")
        try:
            content = scraper()
            results[name] = content
            print(f"    ✓ {len(content)} chars")
        except Exception as e:
            results[name] = f"Source: {name}\nDate: {DATE_STR}\n\nFailed: {e}"
            print(f"    ✗ {e}")
        time.sleep(1)  # Be polite

    # Playwright-based sources (TLDR AI + The Batch)
    pw_results = scrape_playwright_sources()
    results.update(pw_results)

    return results


# ──────────────────────────────────────────────
# AI ANALYSIS — Gemini (primary) → Groq (fallback)
# ──────────────────────────────────────────────
def build_prompt(scraped_data, topic_info, is_groq_fallback=False):
    """Build the analysis prompt, filtering sources per topic."""
    relevant_sources = topic_info.get("sources", list(scraped_data.keys()))

    all_content = ""
    for source_name in relevant_sources:
        if source_name in scraped_data:
            all_content += f"\n\n===== {source_name} =====\n{scraped_data[source_name]}"

    max_chars = 12000 if is_groq_fallback else 50000
    if len(all_content) > max_chars:
        all_content = all_content[:max_chars] + "\n\n[... truncated ...]"

    questions_text = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(topic_info["questions"]))

    return f"""You are a senior AI research analyst writing a weekly intelligence briefing for a technical audience.

TODAY'S DATE: {DATE_STR}

TOPIC: {topic_info['title']}

KEY RESEARCH QUESTIONS:
{questions_text}

Below is data scraped from AI research sources (arXiv, HuggingFace, PubMed, BioRxiv, newsletters, Reddit). Analyze this content and write a detailed research intelligence report.

SCRAPED DATA:
{all_content}

INSTRUCTIONS:
- Write a structured report with clear sections addressing each research question
- Cite specific papers by title and authors when available
- Highlight breakthrough results, novel methods, and significant benchmarks
- Synthesize across sources — identify patterns and emerging themes
- Note gaps — what important questions the data doesn't answer
- Be analytical and opinionated, not just descriptive
- If data is thin on a topic, say so honestly and provide your best analysis
- Use markdown formatting: ## for sections, **bold** for emphasis, - for bullets
- Aim for 600-900 words total

Write the report now:"""


def analyze_with_gemini(prompt):
    """Call Gemini 2.5 Flash API. Returns (text, model_name) or raises on failure."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 8192,
        },
    }

    for attempt in range(3):
        resp = http_requests.post(url, json=payload, timeout=90)
        if resp.status_code == 200:
            data = resp.json()
            parts = data["candidates"][0]["content"]["parts"]
            text = parts[-1]["text"]
            return text, "Gemini 2.5 Flash"
        elif resp.status_code == 429:
            wait = 15 * (attempt + 1)
            print(f"    Gemini rate limited — waiting {wait}s (attempt {attempt+1}/3)...")
            time.sleep(wait)
        else:
            raise Exception(f"Gemini API error {resp.status_code}: {resp.text[:200]}")

    raise Exception("Gemini rate limited after 3 retries")


def analyze_with_groq(prompt):
    """Call Groq API with model fallback. Returns (text, model_name) or raises."""
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
    # 1. Try Gemini with full context
    if GEMINI_API_KEY:
        try:
            prompt = build_prompt(scraped_data, topic_info, is_groq_fallback=False)
            text, model = analyze_with_gemini(prompt)
            print(f"    [{model}] ✓")
            return text, model
        except Exception as e:
            print(f"    Gemini failed: {e} — falling back to Groq...")

    # 2. Fallback to Groq with truncated context
    if GROQ_API_KEY:
        try:
            prompt = build_prompt(scraped_data, topic_info, is_groq_fallback=True)
            text, model = analyze_with_groq(prompt)
            print(f"    [{model}] ✓")
            return text, model
        except Exception as e:
            print(f"    Groq failed: {e}")

    return "Analysis unavailable — all AI providers failed. Will retry next run.", "none"


def run_all_analyses(scraped_data):
    """Run AI analysis for all 7 topics."""
    analyses = {}
    models_used = set()
    for topic_id, topic_info in RESEARCH_TOPICS.items():
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
    """Write the weekly report as a single JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data = {
        "date": DATE_STR,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": SOURCE_NAMES,
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

    for topic_id, topic_info in RESEARCH_TOPICS.items():
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
    """Update research-data/index.json manifest."""
    manifest_path = Path(output_dir) / "index.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"weeks": []}

    existing_dates = {w["date"] for w in manifest["weeks"]}
    if DATE_STR not in existing_dates:
        manifest["weeks"].append({
            "date": DATE_STR,
            "file": f"{DATE_STR}.json",
            "topics": len(RESEARCH_TOPICS),
        })
        manifest["weeks"].sort(key=lambda w: w["date"], reverse=True)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Manifest updated: {len(manifest['weeks'])} weeks total")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AI Research Scraper & Analyzer")
    parser.add_argument("--output-dir", default="docs/research-data", help="Output directory for JSON files")
    args = parser.parse_args()

    print("=" * 55)
    print(f"  AI RESEARCH ANALYZER — {DATE_STR}")
    print("=" * 55)

    # Step 1: Scrape
    print("\n[1/3] Scraping 8 sources...")
    scraped = scrape_all_sources()
    total = sum(len(v) for v in scraped.values())
    print(f"  Total: {total:,} chars from {len(scraped)} sources")

    # Step 2: AI Analysis
    print(f"\n[2/3] AI analysis ({len(RESEARCH_TOPICS)} topics)...")
    analyses = run_all_analyses(scraped)

    # Step 3: Save JSON
    print("\n[3/3] Saving JSON...")
    save_weekly_json(scraped, analyses, args.output_dir)
    update_manifest(args.output_dir)

    print(f"\n{'=' * 55}")
    print("  DONE!")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
