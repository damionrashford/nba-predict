"""
NBA Social Sentiment Collector — Reddit + Mastodon
====================================================
Collects social media posts mentioning NBA teams, scores them with
VADER sentiment analysis, and aggregates daily per-team metrics.

Sources:
  - Reddit r/nba (hot, new, rising) + all 30 team subreddits
  - Mastodon #NBA + team-specific hashtags (Twitter/X replacement)

Outputs:
  - data/sentiment_raw.csv   — individual scored posts
  - data/sentiment.csv        — aggregated daily per-team metrics

Usage:
    python scripts/collect_sentiment.py                    # collect today
    python scripts/collect_sentiment.py --days 7           # search last 7 days
    python scripts/collect_sentiment.py --aggregate-only   # re-aggregate from raw
"""

import argparse
import csv
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nba_predict.data.social_team_mapping import (
    TEAM_HASHTAGS,
    TEAM_SUBREDDITS,
    extract_team_mentions,
    subreddit_to_team,
)

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_DIR = PROJECT_ROOT / "data"

REDDIT_BASE = "https://www.reddit.com"
REDDIT_HEADERS = {"User-Agent": "nba-sentiment-bot/1.0 (NBA prediction research)"}
REDDIT_DELAY = 1.2  # seconds between requests (~50 req/min)

MASTODON_BASE = "https://mastodon.social/api/v1"
MASTODON_DELAY = 0.2  # generous limit (300 req/5min)

RAW_CSV = DATA_DIR / "sentiment_raw.csv"
AGG_CSV = DATA_DIR / "sentiment.csv"

RAW_COLUMNS = [
    "source", "team", "date", "text_preview", "compound", "pos", "neg", "neu",
    "score", "num_comments", "upvote_ratio", "engagement", "post_id",
]
AGG_COLUMNS = [
    "team", "date", "sentiment_score", "sentiment_volume", "sentiment_std",
    "sentiment_engagement", "sentiment_pos_ratio",
]


# ─── HTTP Helpers ────────────────────────────────────────────────────────────

def _reddit_get(path: str, params: dict | None = None) -> dict | None:
    """Fetch a Reddit JSON endpoint with rate limiting."""
    url = f"{REDDIT_BASE}{path}.json"
    try:
        time.sleep(REDDIT_DELAY)
        resp = requests.get(url, headers=REDDIT_HEADERS, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            print(f"    [429] Rate limited, waiting 10s...")
            time.sleep(10)
            return None
        return None
    except Exception as e:
        print(f"    [ERR] {url}: {e}")
        return None


def _mastodon_get(path: str, params: dict | None = None) -> list | None:
    """Fetch a Mastodon API endpoint."""
    url = f"{MASTODON_BASE}{path}"
    try:
        time.sleep(MASTODON_DELAY)
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception as e:
        print(f"    [ERR] {url}: {e}")
        return None


def _strip_html(html: str) -> str:
    """Remove HTML tags from Mastodon content."""
    return re.sub(r"<[^>]+>", " ", html).strip()


# ─── Scoring ─────────────────────────────────────────────────────────────────

def score_text(analyzer: SentimentIntensityAnalyzer, text: str) -> dict:
    """Score text with VADER. Returns compound, pos, neg, neu."""
    scores = analyzer.polarity_scores(text)
    return {
        "compound": scores["compound"],
        "pos": scores["pos"],
        "neg": scores["neg"],
        "neu": scores["neu"],
    }


# ─── Reddit Collection ──────────────────────────────────────────────────────

def _parse_reddit_posts(data: dict | None, source_label: str,
                        analyzer: SentimentIntensityAnalyzer,
                        default_team: str | None = None) -> list[dict]:
    """Parse Reddit listing response into scored rows."""
    if not data or "data" not in data:
        return []

    rows = []
    for child in data["data"].get("children", []):
        post = child.get("data", {})
        if not post:
            continue

        title = post.get("title", "")
        selftext = post.get("selftext", "")
        text = f"{title} {selftext}".strip()
        if not text:
            continue

        # Score the combined text
        scores = score_text(analyzer, text)

        # Determine team mentions
        if default_team:
            teams = [default_team]
        else:
            teams = extract_team_mentions(text)

        if not teams:
            continue

        created = post.get("created_utc", 0)
        date_str = datetime.fromtimestamp(created, tz=timezone.utc).strftime("%Y-%m-%d")
        post_score = post.get("score", 0)
        comments = post.get("num_comments", 0)
        ratio = post.get("upvote_ratio", 0.5)
        engagement = max(post_score, 1) * max(comments, 1)
        post_id = post.get("id", "")

        for team in teams:
            rows.append({
                "source": source_label,
                "team": team,
                "date": date_str,
                "text_preview": text[:200],
                "compound": scores["compound"],
                "pos": scores["pos"],
                "neg": scores["neg"],
                "neu": scores["neu"],
                "score": post_score,
                "num_comments": comments,
                "upvote_ratio": ratio,
                "engagement": engagement,
                "post_id": post_id,
            })

    return rows


def collect_reddit_nba(analyzer: SentimentIntensityAnalyzer,
                       time_filter: str = "day") -> list[dict]:
    """Collect from r/nba hot, new, and rising."""
    print("  [Reddit] r/nba...")
    all_rows = []

    for sort in ["hot", "new", "rising"]:
        data = _reddit_get(f"/r/nba/{sort}", {"limit": "100"})
        rows = _parse_reddit_posts(data, f"reddit_nba_{sort}", analyzer)
        all_rows.extend(rows)
        print(f"    {sort}: {len(rows)} team mentions")

    return all_rows


def collect_reddit_team_subs(analyzer: SentimentIntensityAnalyzer) -> list[dict]:
    """Collect from all 30 team subreddits."""
    print("  [Reddit] Team subreddits...")
    all_rows = []

    for sub, team_code in TEAM_SUBREDDITS.items():
        data = _reddit_get(f"/r/{sub}/hot", {"limit": "50"})
        rows = _parse_reddit_posts(
            data, f"reddit_{sub}", analyzer, default_team=team_code
        )
        all_rows.extend(rows)

        if len(all_rows) % 100 == 0 and all_rows:
            print(f"    ...{len(all_rows)} posts scored")

    print(f"    Total from team subs: {len(all_rows)} posts")
    return all_rows


def collect_reddit_search(analyzer: SentimentIntensityAnalyzer,
                          time_filter: str = "day") -> list[dict]:
    """Search r/nba for each team name."""
    print("  [Reddit] Team search queries...")
    all_rows = []

    # Search for major teams by nickname (avoids ambiguity)
    search_terms = {
        "ATL": "Hawks", "BOS": "Celtics", "BRK": "Nets", "CHI": "Bulls",
        "CHO": "Hornets", "CLE": "Cavaliers", "DAL": "Mavericks",
        "DEN": "Nuggets", "DET": "Pistons", "GSW": "Warriors",
        "HOU": "Rockets", "IND": "Pacers", "LAC": "Clippers",
        "LAL": "Lakers", "MEM": "Grizzlies", "MIA": "Heat",
        "MIL": "Bucks", "MIN": "Timberwolves", "NOP": "Pelicans",
        "NYK": "Knicks", "OKC": "Thunder", "ORL": "Magic",
        "PHI": "76ers", "PHO": "Suns", "POR": "Blazers",
        "SAC": "Kings", "SAS": "Spurs", "TOR": "Raptors",
        "UTA": "Jazz", "WAS": "Wizards",
    }

    for team_code, term in search_terms.items():
        data = _reddit_get("/r/nba/search", {
            "q": term, "restrict_sr": "1", "sort": "new",
            "t": time_filter, "limit": "25",
        })
        rows = _parse_reddit_posts(
            data, "reddit_nba_search", analyzer, default_team=team_code
        )
        all_rows.extend(rows)

    print(f"    Total from search: {len(all_rows)} posts")
    return all_rows


# ─── Mastodon Collection ────────────────────────────────────────────────────

def collect_mastodon(analyzer: SentimentIntensityAnalyzer) -> list[dict]:
    """Collect from Mastodon hashtag timelines for all teams."""
    print("  [Mastodon] Hashtag timelines...")
    all_rows = []

    # General NBA hashtag
    statuses = _mastodon_get("/timelines/tag/nba", {"limit": "40"})
    if statuses:
        for status in statuses:
            text = _strip_html(status.get("content", ""))
            if not text:
                continue
            teams = extract_team_mentions(text)
            if not teams:
                continue

            scores = score_text(analyzer, text)
            created = status.get("created_at", "")[:10]
            favs = status.get("favourites_count", 0)
            reblogs = status.get("reblogs_count", 0)
            engagement = favs + reblogs

            for team in teams:
                all_rows.append({
                    "source": "mastodon_nba",
                    "team": team,
                    "date": created,
                    "text_preview": text[:200],
                    "compound": scores["compound"],
                    "pos": scores["pos"],
                    "neg": scores["neg"],
                    "neu": scores["neu"],
                    "score": favs,
                    "num_comments": status.get("replies_count", 0),
                    "upvote_ratio": 0.5,
                    "engagement": max(engagement, 1),
                    "post_id": status.get("id", ""),
                })

    # Team-specific hashtags
    for team_code, tags in TEAM_HASHTAGS.items():
        for tag in tags:
            statuses = _mastodon_get(f"/timelines/tag/{tag}", {"limit": "40"})
            if not statuses:
                continue
            for status in statuses:
                text = _strip_html(status.get("content", ""))
                if not text:
                    continue

                scores = score_text(analyzer, text)
                created = status.get("created_at", "")[:10]
                favs = status.get("favourites_count", 0)
                reblogs = status.get("reblogs_count", 0)

                all_rows.append({
                    "source": f"mastodon_{tag}",
                    "team": team_code,
                    "date": created,
                    "text_preview": text[:200],
                    "compound": scores["compound"],
                    "pos": scores["pos"],
                    "neg": scores["neg"],
                    "neu": scores["neu"],
                    "score": favs,
                    "num_comments": status.get("replies_count", 0),
                    "upvote_ratio": 0.5,
                    "engagement": max(favs + reblogs, 1),
                    "post_id": status.get("id", ""),
                })

    print(f"    Total from Mastodon: {len(all_rows)} posts")
    return all_rows


# ─── CSV I/O ─────────────────────────────────────────────────────────────────

def append_raw(rows: list[dict], filepath: Path) -> None:
    """Append scored posts to the raw CSV."""
    if not rows:
        return

    file_exists = filepath.exists() and filepath.stat().st_size > 0
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def deduplicate_raw(filepath: Path) -> int:
    """Remove duplicate posts from raw CSV based on (source, post_id, team)."""
    if not filepath.exists():
        return 0

    import pandas as pd
    df = pd.read_csv(filepath)
    before = len(df)
    df = df.drop_duplicates(subset=["source", "post_id", "team"], keep="last")
    df.to_csv(filepath, index=False)
    return before - len(df)


def aggregate_daily(raw_path: Path, output_path: Path) -> None:
    """Aggregate raw scored posts into daily per-team sentiment metrics."""
    import pandas as pd

    if not raw_path.exists():
        print("  [SKIP] No raw sentiment data to aggregate.")
        return

    df = pd.read_csv(raw_path)
    if df.empty:
        print("  [SKIP] Raw sentiment CSV is empty.")
        return

    df["compound"] = pd.to_numeric(df["compound"], errors="coerce")
    df["engagement"] = pd.to_numeric(df["engagement"], errors="coerce").fillna(1)

    # Aggregate by (team, date)
    agg = df.groupby(["team", "date"]).agg(
        sentiment_score=("compound", "mean"),
        sentiment_volume=("compound", "count"),
        sentiment_std=("compound", "std"),
        sentiment_engagement=("engagement", "mean"),
        sentiment_pos_ratio=("compound", lambda x: (x > 0.05).mean()),
    ).reset_index()

    # Fill NaN std (single-post days) with 0
    agg["sentiment_std"] = agg["sentiment_std"].fillna(0)

    agg.to_csv(output_path, index=False)
    print(f"  [DONE] Aggregated: {len(agg)} team-date rows → {output_path.name}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NBA Social Sentiment Collector")
    parser.add_argument("--days", type=int, default=1,
                        help="Days of history to search (default: 1)")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip collection, just re-aggregate from raw")
    parser.add_argument("--skip-mastodon", action="store_true",
                        help="Skip Mastodon collection")
    parser.add_argument("--skip-search", action="store_true",
                        help="Skip Reddit team search queries (faster)")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    if args.aggregate_only:
        print("Re-aggregating from raw data...")
        aggregate_daily(RAW_CSV, AGG_CSV)
        return

    time_filter = "day"
    if args.days > 7:
        time_filter = "month"
    elif args.days > 1:
        time_filter = "week"

    print(f"\nNBA Social Sentiment Collector")
    print(f"{'=' * 50}")
    print(f"Time filter: {time_filter}")
    print(f"Output: {DATA_DIR}/")
    print(f"{'=' * 50}")

    analyzer = SentimentIntensityAnalyzer()
    all_rows = []
    start = time.time()

    # Phase 1: Reddit r/nba
    rows = collect_reddit_nba(analyzer, time_filter)
    all_rows.extend(rows)

    # Phase 2: Reddit team subreddits
    rows = collect_reddit_team_subs(analyzer)
    all_rows.extend(rows)

    # Phase 3: Reddit search (optional)
    if not args.skip_search:
        rows = collect_reddit_search(analyzer, time_filter)
        all_rows.extend(rows)

    # Phase 4: Mastodon
    if not args.skip_mastodon:
        rows = collect_mastodon(analyzer)
        all_rows.extend(rows)

    elapsed = time.time() - start
    print(f"\n  Collected {len(all_rows)} total posts in {elapsed:.0f}s")

    # Write raw
    append_raw(all_rows, RAW_CSV)
    removed = deduplicate_raw(RAW_CSV)
    if removed:
        print(f"  Removed {removed} duplicate posts")

    # Aggregate
    aggregate_daily(RAW_CSV, AGG_CSV)

    # Summary
    import pandas as pd
    if AGG_CSV.exists():
        agg = pd.read_csv(AGG_CSV)
        teams_with_data = agg["team"].nunique()
        dates_with_data = agg["date"].nunique()
        avg_volume = agg["sentiment_volume"].mean()
        avg_sentiment = agg["sentiment_score"].mean()
        print(f"\n  Summary:")
        print(f"    Teams with data: {teams_with_data}")
        print(f"    Dates covered:   {dates_with_data}")
        print(f"    Avg posts/team/day: {avg_volume:.1f}")
        print(f"    Avg sentiment:   {avg_sentiment:+.3f}")

    print(f"\n{'=' * 50}")
    print(f"COMPLETE — {elapsed:.0f}s")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
