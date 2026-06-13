#!/usr/bin/env python3
"""
Podcast digest pipeline — runs on GitHub Actions.
1. Fetches latest episodes from RSS feeds
2. Transcribes audio via Groq Whisper API (free)
3. Summarizes via Groq LLM (free)
4. Writes index.html for GitHub Pages
"""

import os, json, re, tempfile, urllib.request
from pathlib import Path
from datetime import datetime
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

SEEN_FILE  = Path("seen_episodes.json")
SUMMARY_DIR = Path("summaries")
SUMMARY_DIR.mkdir(exist_ok=True)

# ── Podcast sources ────────────────────────────────────────────────────────────

PODCASTS = [
    {
        "name": "All In Podcast",
        "rss": "https://rss.libsyn.com/shows/254861/destinations/1928300.xml",
        "max": 3,
    },
    {
        "name": "BG2 Pod",
        "rss": "https://feeds.simplecast.com/bg2pod",
        "max": 2,
    },
    {
        "name": "20VC",
        "rss": "https://feeds.simplecast.com/9OFBsQJf",
        "max": 2,
    },
    {
        "name": "Invest Like the Best",
        "rss": "https://feeds.simplecast.com/wXSGnlMp",
        "max": 2,
    },
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:80]

def load_seen():
    return set(json.loads(SEEN_FILE.read_text())) if SEEN_FILE.exists() else set()

def save_seen(seen):
    SEEN_FILE.write_text(json.dumps(sorted(seen), indent=2))

# ── Transcription via Groq Whisper API ────────────────────────────────────────

def transcribe(audio_url: str, title: str) -> str:
    print(f"  [transcribe] downloading audio...")
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        urllib.request.urlretrieve(audio_url, tmp.name)
        tmp_path = tmp.name

    print(f"  [transcribe] sending to Groq Whisper...")
    with open(tmp_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(title[:40] + ".mp3", f),
            model="whisper-large-v3",
            response_format="text",
        )
    os.unlink(tmp_path)
    return result

# ── Summarization via Groq LLM ────────────────────────────────────────────────

PROMPT = """You are a VC analyst at a top-tier venture firm.
Summarize this podcast transcript for a junior analyst who needs to be briefed quickly.

Structure your response EXACTLY as follows (use these headers verbatim):

## TL;DR
2-3 sentence executive summary of the entire episode.

## Key Themes
Bullet list of 4-6 major topics discussed, each with 1-2 sentences of context.

## VC Signals & Market Moves
Bullet list of specific investment theses, sector calls, fund moves, or market opinions expressed. Include who said what.

## Notable Quotes
3-5 direct quotes that best capture the episode's ideas. Format as: "Quote" — Speaker

## Vocabulary & Concepts to Know
Any jargon, fund names, company names, or concepts a new analyst should look up. One-line definition each.

---
TRANSCRIPT:
{transcript}
"""

def summarize(title: str, transcript: str) -> str:
    slug = slugify(title)
    cache = SUMMARY_DIR / f"{slug}.json"
    if cache.exists():
        return json.loads(cache.read_text())["summary"]

    print(f"  [summarize] calling Groq LLM...")
    trunc = transcript[:60000]
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=2000,
        messages=[{"role": "user", "content": PROMPT.format(transcript=trunc)}],
    )
    summary = resp.choices[0].message.content
    cache.write_text(json.dumps({"title": title, "summary": summary}, indent=2))
    return summary

# ── RSS fetching ───────────────────────────────────────────────────────────────

def fetch_episodes(seen: set) -> list[dict]:
    import feedparser, email.utils
    episodes = []

    for pod in PODCASTS:
        print(f"\n[{pod['name']}] checking RSS...")
        feed = feedparser.parse(pod["rss"])
        if not feed.entries:
            print("  no entries found")
            continue

        for entry in feed.entries[:pod["max"]]:
            ep_id = entry.get("id") or entry.get("link") or entry.get("title")
            title = entry.get("title", "untitled")

            if ep_id in seen:
                print(f"  [skip] {title[:60]}")
                continue

            # find audio URL
            audio_url = None
            for enc in entry.get("enclosures", []):
                if "audio" in enc.get("type", "") or enc.get("href","").endswith(".mp3"):
                    audio_url = enc.get("href") or enc.get("url")
                    break
            if not audio_url:
                print(f"  [skip] no audio: {title[:60]}")
                continue

            try:
                dt = email.utils.parsedate_to_datetime(entry.get("published", ""))
                date_str = dt.strftime("%Y-%m-%d")
            except:
                date_str = datetime.now().strftime("%Y-%m-%d")

            print(f"  [new] {title[:60]}")
            try:
                transcript = transcribe(audio_url, title)
                summary    = summarize(title, transcript)
                seen.add(ep_id)
                episodes.append({
                    "title": title,
                    "date": date_str,
                    "source": pod["name"],
                    "summary": summary,
                })
            except Exception as e:
                print(f"  [error] {e}")

    return episodes

# ── Web page ───────────────────────────────────────────────────────────────────

def md_to_html(md: str) -> str:
    lines, out, in_ul = md.split("\n"), [], False
    for line in lines:
        if line.startswith("## "):
            if in_ul: out.append("</ul>"); in_ul = False
            out.append(f'<h3>{line[3:]}</h3>')
        elif line.startswith("- ") or line.startswith("* "):
            if not in_ul: out.append("<ul>"); in_ul = True
            inner = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line[2:])
            out.append(f"<li>{inner}</li>")
        elif line.startswith("---"):
            if in_ul: out.append("</ul>"); in_ul = False
        elif line.strip() == "":
            if in_ul: out.append("</ul>"); in_ul = False
        else:
            if in_ul: out.append("</ul>"); in_ul = False
            inner = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
            out.append(f"<p>{inner}</p>")
    if in_ul: out.append("</ul>")
    return "\n".join(out)

def build_page(all_summaries: list[dict]):
    cards = ""
    for ep in sorted(all_summaries, key=lambda e: e["date"], reverse=True):
        cards += f"""
        <div class="card">
          <div class="card-header">
            <span class="date">{ep['date']}</span>
            <h2>{ep['title']}</h2>
            <span class="source">{ep.get('source','')}</span>
          </div>
          <div class="card-body">{md_to_html(ep['summary'])}</div>
        </div>"""

    updated = datetime.utcnow().strftime("%b %d %Y, %I:%M %p UTC")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VC Podcast Digest</title>
<style>
  :root {{ --bg:#0f1117;--surface:#1a1d27;--border:#2a2d3a;--text:#e2e8f0;--muted:#8892a4;--accent:#6366f1;--yellow:#fbbf24; }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:15px;line-height:1.6}}
  header{{background:var(--surface);border-bottom:1px solid var(--border);padding:20px 24px;position:sticky;top:0;z-index:10;display:flex;align-items:center;justify-content:space-between}}
  header h1{{font-size:18px;font-weight:700}} header h1 span{{color:var(--accent)}}
  .updated{{font-size:12px;color:var(--muted)}}
  main{{max-width:780px;margin:0 auto;padding:24px 16px 60px}}
  .card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;margin-bottom:20px;overflow:hidden}}
  .card-header{{padding:20px 24px 16px;border-bottom:1px solid var(--border)}}
  .date{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px}}
  .card-header h2{{font-size:17px;font-weight:600;margin:6px 0 4px;line-height:1.3}}
  .source{{font-size:12px;color:var(--accent);font-weight:500}}
  .card-body{{padding:20px 24px}}
  .card-body h3{{font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--accent);margin:20px 0 8px}}
  .card-body h3:first-child{{margin-top:0}}
  .card-body p,.card-body li{{color:var(--text);margin-bottom:6px}}
  .card-body ul{{padding-left:18px;margin-bottom:8px}}
  .card-body strong{{color:var(--yellow)}}
</style>
</head>
<body>
<header>
  <h1>VC Podcast <span>Digest</span></h1>
  <span class="updated">Updated {updated}</span>
</header>
<main>{'<p style="color:var(--muted);text-align:center;padding:60px">No episodes yet.</p>' if not all_summaries else cards}
</main>
</body>
</html>"""
    Path("index.html").write_text(html)
    print(f"\nWrote index.html with {len(all_summaries)} episodes.")

# ── Main ───────────────────────────────────────────────────────────────────────

def load_all_summaries() -> list[dict]:
    out = []
    for f in SUMMARY_DIR.glob("*.json"):
        try:
            out.append(json.loads(f.read_text()))
        except:
            pass
    return out

def main():
    seen = load_seen()
    new_episodes = fetch_episodes(seen)
    save_seen(seen)

    all_summaries = load_all_summaries()
    build_page(all_summaries)

    print(f"\nDone. {len(new_episodes)} new episodes processed.")

if __name__ == "__main__":
    main()
