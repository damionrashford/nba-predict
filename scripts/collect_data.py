"""
NBA Dataset Builder — Basketball Reference
============================================
Parallel data pipeline with rotating proxies.
Produces 16 CSV datasets covering 2000-2026 seasons.

Usage:
    python scripts/collect_data.py              # collect everything
    python scripts/collect_data.py --test       # test with 1 page
    python scripts/collect_data.py --tier 1     # collect only tier 1
"""

import csv
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cloudscraper
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ─── Config ──────────────────────────────────────────────────────────────────

BASE_URL = "https://www.basketball-reference.com"
DATA_DIR = Path(__file__).parent.parent / "data"
YEAR_START = 2000
YEAR_END = 2026  # inclusive (season ending year)
YEARS = list(range(YEAR_START, YEAR_END + 1))

# Workers: concurrent requests
MAX_WORKERS = 10

# Delay between direct requests to stay under rate limit (seconds)
REQUEST_DELAY = 3.5

# Thread-safe rate limiter
import threading
_rate_lock = threading.Lock()
_last_request_time = 0.0

# Proxy sources — Proxifly (validated every 5 min) + TheSpeedX as backup
PROXY_URLS = [
    ("http",   "https://cdn.jsdelivr.net/gh/proxifly/free-proxy-list@main/proxies/protocols/http/data.txt"),
    ("https",  "https://cdn.jsdelivr.net/gh/proxifly/free-proxy-list@main/proxies/protocols/https/data.txt"),
    ("socks5", "https://cdn.jsdelivr.net/gh/proxifly/free-proxy-list@main/proxies/protocols/socks5/data.txt"),
    ("http",   "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt"),
    ("socks5", "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/socks5.txt"),
]

# All 30 NBA teams — using codes that work for current-era URLs
TEAMS = [
    "ATL", "BOS", "BRK", "CHO", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]

# Teams that changed codes over the years (relocations / rebrands)
# BBRef codes: CHH = original Charlotte Hornets, CHA = Bobcats, CHO = current Hornets
#              NOH = New Orleans Hornets, NOK = NO/OKC Hornets (Katrina), NOP = Pelicans
TEAM_CODE_OVERRIDES = {
    ("BRK", y): "NJN" for y in range(2000, 2013)
} | {
    ("CHO", y): "CHH" for y in range(2000, 2003)   # original Hornets (moved to NO after 2002)
} | {
    ("CHO", y): None for y in range(2003, 2005)     # no Charlotte team 2003-2004
} | {
    ("CHO", y): "CHA" for y in range(2005, 2015)    # Bobcats
} | {
    ("NOP", y): None for y in range(2000, 2003)      # no New Orleans team before 2003
} | {
    ("NOP", y): "NOH" for y in range(2003, 2006)    # New Orleans Hornets
} | {
    ("NOP", y): "NOK" for y in range(2006, 2008)    # NO/OKC Hornets (Katrina)
} | {
    ("NOP", y): "NOH" for y in range(2008, 2014)    # New Orleans Hornets (back home)
} | {
    ("OKC", y): "SEA" for y in range(2000, 2009)
} | {
    ("MEM", y): "VAN" for y in range(2000, 2002)
}

# Tier 1: Player stats pages (league-wide, 1 page = all players)
PLAYER_STAT_PAGES = {
    "players_per_game":     ("per_game",     "per_game_stats"),
    "players_totals":       ("totals",       "totals_stats"),
    "players_per_36":       ("per_minute",   "per_minute_stats"),
    "players_per_100":      ("per_poss",     "per_poss"),
    "players_advanced":     ("advanced",     "advanced"),
    "players_shooting":     ("shooting",     "shooting"),
    "players_adj_shooting": ("adj_shooting", "adj_shooting"),
    "players_play_by_play": ("play-by-play", "pbp_stats"),
}

# Tier 2: Team stats tables embedded in season main page
TEAM_STAT_TABLES = {
    "teams_per_game":  "per_game-team",
    "teams_opponent":  "per_game-opponent",
    "teams_advanced":  "advanced-team",
    "teams_shooting":  "shooting-team",
}


# ─── Proxy Pool ──────────────────────────────────────────────────────────────

class ProxyPool:
    """Loads and rotates through free proxies."""

    def __init__(self):
        self.proxies = []
        self.working = []
        self.failed = set()

    def load(self):
        """Pull proxy lists from GitHub (Proxifly + TheSpeedX)."""
        print("[*] Loading proxy lists...")
        seen = set()
        all_proxies = []

        for proto, url in PROXY_URLS:
            try:
                resp = requests.get(url, timeout=10)
                lines = resp.text.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    if line and ":" in line:
                        if proto == "socks5":
                            addr = f"socks5://{line}"
                        else:
                            addr = f"http://{line}"
                        if addr not in seen:
                            seen.add(addr)
                            all_proxies.append(addr)
            except Exception as e:
                print(f"  [!] Failed to load {proto} from {url.split('/')[2]}: {e}")

        random.shuffle(all_proxies)
        self.proxies = all_proxies
        print(f"  [+] Loaded {len(self.proxies)} unique proxies")

    def get(self):
        """Get a random proxy, preferring known-working ones."""
        if self.working:
            return random.choice(self.working)
        available = [p for p in self.proxies if p not in self.failed]
        if not available:
            self.failed.clear()
            available = self.proxies
        return random.choice(available) if available else None

    def mark_good(self, proxy):
        if proxy and proxy not in self.working:
            self.working.append(proxy)
            if len(self.working) > 100:
                self.working = self.working[-100:]

    def mark_bad(self, proxy):
        if proxy:
            self.failed.add(proxy)
            if proxy in self.working:
                self.working.remove(proxy)


# ─── HTTP Session ────────────────────────────────────────────────────────────

_thread_local = threading.local()


def create_session():
    """Create a browser-like session that handles Cloudflare challenges."""
    session = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "darwin", "desktop": True},
        delay=2,
    )
    session.headers.update({
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    })
    return session


def get_session():
    """Get a thread-local session (reuses across calls in the same thread)."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = create_session()
    return _thread_local.session


# ─── Fetch with rate-limited direct requests ─────────────────────────────────

def _wait_for_rate_limit():
    """Thread-safe rate limiter: ensures REQUEST_DELAY seconds between direct requests."""
    global _last_request_time
    with _rate_lock:
        now = time.time()
        wait = REQUEST_DELAY - (now - _last_request_time)
        if wait > 0:
            time.sleep(wait)
        _last_request_time = time.time()


def fetch_page(url, proxy_pool, max_retries=3):
    """
    Download a page with rate-limited direct requests.
    Strategy: polite direct request (3.5s spacing) → quick proxy on 429 → retry.
    Returns HTML string or None on failure.
    """
    session = get_session()

    for attempt in range(max_retries):
        # Rate-limited direct request (all threads queue here)
        _wait_for_rate_limit()

        try:
            resp = session.get(url, timeout=20)
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code == 404:
                return None
            elif resp.status_code == 429:
                # Quick single proxy attempt (short timeout, no retry loop)
                proxy = proxy_pool.get()
                if proxy:
                    try:
                        resp = session.get(
                            url,
                            proxies={"http": proxy, "https": proxy},
                            timeout=8,
                        )
                        if resp.status_code == 200:
                            proxy_pool.mark_good(proxy)
                            return resp.text
                        elif resp.status_code == 404:
                            return None
                        else:
                            proxy_pool.mark_bad(proxy)
                    except Exception:
                        proxy_pool.mark_bad(proxy)
                # Back off before retrying direct
                time.sleep(5)
                continue
            else:
                # Other error (5xx etc) — retry after brief pause
                time.sleep(2)
                continue
        except Exception:
            time.sleep(2)
            continue

    return None


# ─── HTML Parsing ────────────────────────────────────────────────────────────

def parse_table_from_html(html, table_id, season=None, extra_cols=None):
    """
    Extract a specific table from a basketball-reference HTML page.
    Strips HTML comments to expose hidden tables.
    Returns list of dicts (rows).
    """
    if not html:
        return []

    # Strip HTML comments to expose hidden tables
    html = html.replace("<!--", "").replace("-->", "")
    soup = BeautifulSoup(html, "lxml")

    table = soup.find("table", {"id": table_id})
    if not table:
        # Try partial match
        for t in soup.find_all("table"):
            tid = t.get("id", "")
            if table_id in tid:
                table = t
                break

    if not table:
        return []

    rows = []
    thead = table.find("thead")
    headers = []
    if thead:
        header_rows = thead.find_all("tr")
        last_header = header_rows[-1] if header_rows else None
        if last_header:
            for th in last_header.find_all(["th", "td"]):
                stat = th.get("data-stat", th.get_text(strip=True))
                headers.append(stat)

    tbody = table.find("tbody")
    if not tbody:
        return []

    for tr in tbody.find_all("tr"):
        if tr.get("class") and "thead" in tr.get("class", []):
            continue
        if tr.find("th", {"scope": "col"}):
            continue

        cells = tr.find_all(["th", "td"])
        if not cells:
            continue

        row = {}
        for i, cell in enumerate(cells):
            key = cell.get("data-stat", headers[i] if i < len(headers) else f"col_{i}")
            text = cell.get_text(strip=True)
            row[key] = text

        if not any(v for v in row.values()):
            continue

        if season:
            row["season"] = season
        if extra_cols:
            row.update(extra_cols)

        rows.append(row)

    return rows


def parse_multiple_tables(html, table_ids, season=None):
    """Extract multiple tables from the same HTML page. Returns dict of {name: rows}."""
    results = {}
    for name, table_id in table_ids.items():
        results[name] = parse_table_from_html(html, table_id, season=season)
    return results


# ─── CSV Writing ─────────────────────────────────────────────────────────────

def append_to_csv(filepath, rows):
    """Append rows (list of dicts) to a CSV file. Creates file with headers if new.
    If new columns appear, rewrites the entire file with the merged header."""
    if not rows:
        return

    filepath = Path(filepath)
    file_exists = filepath.exists() and filepath.stat().st_size > 0

    # Collect all column names from the new rows
    new_keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                new_keys.append(k)
                seen.add(k)

    if not file_exists:
        # Simple case: write new file
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=new_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return

    # Read existing data
    existing_rows = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        existing_headers = list(reader.fieldnames or [])
        existing_rows = list(reader)

    # Merge headers (existing first, then any new ones)
    merged_headers = list(existing_headers)
    for k in new_keys:
        if k not in merged_headers:
            merged_headers.append(k)

    has_new_cols = len(merged_headers) > len(existing_headers)

    if has_new_cols:
        # Rewrite entire file with merged header so all rows have consistent width
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=merged_headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerows(rows)
    else:
        # No new columns — safe to append
        with open(filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=merged_headers, extrasaction="ignore")
            writer.writerows(rows)


def get_completed_seasons(filepath):
    """Check which seasons are already in a CSV file (for resume support)."""
    filepath = Path(filepath)
    if not filepath.exists() or filepath.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(filepath, usecols=["season"], dtype={"season": str})
        return set(df["season"].unique())
    except (ValueError, KeyError):
        return set()


def get_completed_team_seasons(filepath):
    """Check which (team, season) pairs exist in a CSV (for roster/schedule resume)."""
    filepath = Path(filepath)
    if not filepath.exists() or filepath.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(filepath, usecols=["team", "season"], dtype=str)
        return set(df["team"] + "_" + df["season"])
    except (ValueError, KeyError):
        return set()


# ─── Data Collection Tasks ───────────────────────────────────────────────────

def collect_player_stats(proxy_pool, executor):
    """Tier 1: Collect all league-wide player stat pages."""
    print(f"\n{'='*60}")
    print("TIER 1: League-Wide Player Stats")
    print(f"{'='*60}")

    for csv_name, (url_slug, table_id) in PLAYER_STAT_PAGES.items():
        csv_path = DATA_DIR / f"{csv_name}.csv"
        done = get_completed_seasons(csv_path)

        years_todo = [y for y in YEARS if str(y) not in done]
        if not years_todo:
            print(f"  [SKIP] {csv_name}.csv — all seasons complete")
            continue

        print(f"\n  [{csv_name}] Collecting {len(years_todo)} seasons...")

        urls = [(y, f"{BASE_URL}/leagues/NBA_{y}_{url_slug}.html") for y in years_todo]

        def fetch_and_parse(args):
            year, url = args
            html = fetch_page(url, proxy_pool)
            if html:
                rows = parse_table_from_html(html, table_id, season=str(year))
                return year, rows
            return year, []

        all_rows = []
        completed = 0
        total = len(urls)

        futures = {executor.submit(fetch_and_parse, u): u for u in urls}
        for future in futures:
            try:
                year, rows = future.result(timeout=60)
                completed += 1
                if rows:
                    all_rows.extend(rows)
                    print(f"    [{completed}/{total}] {year}: {len(rows)} players")
                else:
                    print(f"    [{completed}/{total}] {year}: no data")
            except Exception as e:
                completed += 1
                print(f"    [{completed}/{total}] error: {e}")

        if all_rows:
            append_to_csv(csv_path, all_rows)
            print(f"  [DONE] {csv_name}.csv — {len(all_rows)} total rows")


def collect_team_stats(proxy_pool, executor):
    """Tier 2: Collect team stats from season main pages."""
    print(f"\n{'='*60}")
    print("TIER 2: Team Stats (from season pages)")
    print(f"{'='*60}")

    first_csv = DATA_DIR / "teams_per_game.csv"
    done = get_completed_seasons(first_csv)
    years_todo = [y for y in YEARS if str(y) not in done]

    if not years_todo:
        print("  [SKIP] All team stats complete")
        return

    print(f"  Collecting {len(years_todo)} season pages (4 tables each)...")

    def fetch_and_parse(year):
        url = f"{BASE_URL}/leagues/NBA_{year}.html"
        html = fetch_page(url, proxy_pool)
        if html:
            results = parse_multiple_tables(html, TEAM_STAT_TABLES, season=str(year))
            return year, results
        return year, {}

    completed = 0
    total = len(years_todo)
    all_results = {name: [] for name in TEAM_STAT_TABLES}

    futures = {executor.submit(fetch_and_parse, y): y for y in years_todo}
    for future in futures:
        try:
            year, results = future.result(timeout=60)
            completed += 1
            for name, rows in results.items():
                all_results[name].extend(rows)
            count = sum(len(r) for r in results.values())
            print(f"    [{completed}/{total}] {year}: {count} rows across 4 tables")
        except Exception as e:
            completed += 1
            print(f"    [{completed}/{total}] error: {e}")

    for name, rows in all_results.items():
        if rows:
            csv_path = DATA_DIR / f"{name}.csv"
            append_to_csv(csv_path, rows)
            print(f"  [DONE] {name}.csv — {len(rows)} rows")


def collect_standings(proxy_pool, executor):
    """Collect standings data."""
    print(f"\n{'='*60}")
    print("STANDINGS")
    print(f"{'='*60}")

    csv_path = DATA_DIR / "standings.csv"
    done = get_completed_seasons(csv_path)
    years_todo = [y for y in YEARS if str(y) not in done]

    if not years_todo:
        print("  [SKIP] Standings complete")
        return

    def fetch_and_parse(year):
        url = f"{BASE_URL}/leagues/NBA_{year}_standings.html"
        html = fetch_page(url, proxy_pool)
        if not html:
            return year, []

        # Try expanded_standings first (all 30 teams, works all years)
        rows = parse_table_from_html(html, "expanded_standings", season=str(year))
        if rows:
            return year, rows

        # Fallback: conference standings (newer years)
        all_rows = []
        for conf, table_id in [("EAST", "confs_standings_E"), ("WEST", "confs_standings_W")]:
            conf_rows = parse_table_from_html(
                html, table_id, season=str(year),
                extra_cols={"conference": conf}
            )
            all_rows.extend(conf_rows)
        return year, all_rows

    all_rows = []
    completed = 0
    total = len(years_todo)

    futures = {executor.submit(fetch_and_parse, y): y for y in years_todo}
    for future in futures:
        try:
            year, rows = future.result(timeout=60)
            completed += 1
            all_rows.extend(rows)
            print(f"    [{completed}/{total}] {year}: {len(rows)} teams")
        except Exception as e:
            completed += 1
            print(f"    [{completed}/{total}] error: {e}")

    if all_rows:
        append_to_csv(csv_path, all_rows)
        print(f"  [DONE] standings.csv — {len(all_rows)} rows")


def _get_team_code(team, year):
    """Get the correct team code for a given year (handles relocations)."""
    return TEAM_CODE_OVERRIDES.get((team, year), team)


def collect_rosters(proxy_pool, executor):
    """Tier 3a: Collect rosters from team pages."""
    print(f"\n{'='*60}")
    print("TIER 3a: Rosters")
    print(f"{'='*60}")

    csv_path = DATA_DIR / "rosters.csv"
    done = get_completed_team_seasons(csv_path)

    jobs = []
    for team in TEAMS:
        for year in YEARS:
            code = _get_team_code(team, year)
            if code is None:
                continue  # franchise didn't exist this year
            season_key = f"{team}_{year}"
            if season_key not in done:
                jobs.append((team, year, code))

    if not jobs:
        print("  [SKIP] All rosters complete")
        return

    print(f"  Collecting {len(jobs)} team-seasons...")

    def fetch_and_parse(args):
        team, year, code = args
        url = f"{BASE_URL}/teams/{code}/{year}.html"
        html = fetch_page(url, proxy_pool)
        if html:
            rows = parse_table_from_html(
                html, "roster", season=str(year),
                extra_cols={"team": team}
            )
            return team, year, rows
        return team, year, []

    all_rows = []
    completed = 0
    total = len(jobs)

    futures = {executor.submit(fetch_and_parse, j): j for j in jobs}
    for future in futures:
        try:
            team, year, rows = future.result(timeout=60)
            completed += 1
            all_rows.extend(rows)
            if completed % 50 == 0 or completed == total:
                print(f"    [{completed}/{total}] Latest: {team} {year} ({len(rows)} players)")
        except Exception as e:
            completed += 1
            if completed % 50 == 0:
                print(f"    [{completed}/{total}] error: {e}")

    if all_rows:
        append_to_csv(csv_path, all_rows)
        print(f"  [DONE] rosters.csv — {len(all_rows)} rows")


def collect_schedules(proxy_pool, executor):
    """Tier 3b: Collect game schedules from team pages."""
    print(f"\n{'='*60}")
    print("TIER 3b: Schedules / Game Results")
    print(f"{'='*60}")

    csv_path = DATA_DIR / "schedules.csv"
    done = get_completed_team_seasons(csv_path)

    jobs = []
    for team in TEAMS:
        for year in YEARS:
            code = _get_team_code(team, year)
            if code is None:
                continue  # franchise didn't exist this year
            season_key = f"{team}_{year}"
            if season_key not in done:
                jobs.append((team, year, code))

    if not jobs:
        print("  [SKIP] All schedules complete")
        return

    print(f"  Collecting {len(jobs)} team-season schedules...")

    def fetch_and_parse(args):
        team, year, code = args
        url = f"{BASE_URL}/teams/{code}/{year}_games.html"
        html = fetch_page(url, proxy_pool)
        if html:
            rows = parse_table_from_html(
                html, "games", season=str(year),
                extra_cols={"team": team}
            )
            return team, year, rows
        return team, year, []

    all_rows = []
    completed = 0
    total = len(jobs)

    futures = {executor.submit(fetch_and_parse, j): j for j in jobs}
    for future in futures:
        try:
            team, year, rows = future.result(timeout=60)
            completed += 1
            all_rows.extend(rows)
            if completed % 50 == 0 or completed == total:
                print(f"    [{completed}/{total}] Latest: {team} {year} ({len(rows)} games)")
        except Exception as e:
            completed += 1
            if completed % 50 == 0:
                print(f"    [{completed}/{total}] error: {e}")

    if all_rows:
        append_to_csv(csv_path, all_rows)
        print(f"  [DONE] schedules.csv — {len(all_rows)} rows")


def collect_draft(proxy_pool, executor):
    """Tier 4a: Collect draft data."""
    print(f"\n{'='*60}")
    print("TIER 4a: Draft")
    print(f"{'='*60}")

    csv_path = DATA_DIR / "draft.csv"
    done = get_completed_seasons(csv_path)
    draft_years = [y for y in range(YEAR_START, YEAR_END) if str(y) not in done]

    if not draft_years:
        print("  [SKIP] All drafts complete")
        return

    print(f"  Collecting {len(draft_years)} drafts...")

    def fetch_and_parse(year):
        url = f"{BASE_URL}/draft/NBA_{year}.html"
        html = fetch_page(url, proxy_pool)
        if html:
            rows = parse_table_from_html(html, "stats", season=str(year))
            if not rows:
                rows = parse_table_from_html(html, "draft_stats", season=str(year))
            return year, rows
        return year, []

    all_rows = []
    completed = 0
    total = len(draft_years)

    futures = {executor.submit(fetch_and_parse, y): y for y in draft_years}
    for future in futures:
        try:
            year, rows = future.result(timeout=60)
            completed += 1
            all_rows.extend(rows)
            print(f"    [{completed}/{total}] {year}: {len(rows)} picks")
        except Exception as e:
            completed += 1
            print(f"    [{completed}/{total}] error: {e}")

    if all_rows:
        append_to_csv(csv_path, all_rows)
        print(f"  [DONE] draft.csv — {len(all_rows)} rows")


def collect_awards(proxy_pool, executor):
    """Tier 4b: Collect awards data."""
    print(f"\n{'='*60}")
    print("TIER 4b: Awards")
    print(f"{'='*60}")

    csv_path = DATA_DIR / "awards.csv"
    done = get_completed_seasons(csv_path)
    years_todo = [y for y in YEARS if str(y) not in done]

    if not years_todo:
        print("  [SKIP] All awards complete")
        return

    print(f"  Collecting {len(years_todo)} award pages...")

    def fetch_and_parse(year):
        url = f"{BASE_URL}/awards/awards_{year}.html"
        html = fetch_page(url, proxy_pool)
        if not html:
            return year, []

        all_award_rows = []
        html_clean = html.replace("<!--", "").replace("-->", "")
        soup = BeautifulSoup(html_clean, "lxml")

        for table in soup.find_all("table"):
            table_id = table.get("id", "unknown")
            rows = parse_table_from_html(
                html, table_id, season=str(year),
                extra_cols={"award_type": table_id}
            )
            all_award_rows.extend(rows)

        return year, all_award_rows

    all_rows = []
    completed = 0
    total = len(years_todo)

    futures = {executor.submit(fetch_and_parse, y): y for y in years_todo}
    for future in futures:
        try:
            year, rows = future.result(timeout=60)
            completed += 1
            all_rows.extend(rows)
            print(f"    [{completed}/{total}] {year}: {len(rows)} entries")
        except Exception as e:
            completed += 1
            print(f"    [{completed}/{total}] error: {e}")

    if all_rows:
        append_to_csv(csv_path, all_rows)
        print(f"  [DONE] awards.csv — {len(all_rows)} rows")


# ─── Test Mode ───────────────────────────────────────────────────────────────

def test_run(proxy_pool):
    """Quick test: fetch 1 page, parse it, print results."""
    print("\n[TEST MODE] Fetching a single page to verify everything works...")

    url = f"{BASE_URL}/leagues/NBA_2025_per_game.html"
    print(f"  URL: {url}")

    start = time.time()
    html = fetch_page(url, proxy_pool)
    elapsed = time.time() - start

    if not html:
        print(f"  [FAIL] Could not fetch page ({elapsed:.1f}s)")
        print("  Trying direct (no proxy)...")
        session = get_session()
        resp = session.get(url, timeout=25)
        if resp.status_code == 200:
            html = resp.text
            print(f"  [OK] Direct request worked ({resp.status_code})")
        else:
            print(f"  [FAIL] Direct request also failed ({resp.status_code})")
            return False

    print(f"  [OK] Retrieved {len(html)} bytes in {elapsed:.1f}s")

    rows = parse_table_from_html(html, "per_game_stats", season="2025")
    print(f"  [OK] Parsed {len(rows)} player rows")

    if rows:
        for row in rows[:3]:
            name = row.get("name_display", row.get("player", "?"))
            pts = row.get("pts_per_g", row.get("pts", "?"))
            team = row.get("team_name_abbr", row.get("team_id", "?"))
            print(f"       {name} ({team}): {pts} PPG")

    print(f"\n  [OK] Test passed! Ready to collect data.")
    return True


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    DATA_DIR.mkdir(exist_ok=True)

    test_mode = "--test" in sys.argv
    tier_only = None
    if "--tier" in sys.argv:
        idx = sys.argv.index("--tier")
        if idx + 1 < len(sys.argv):
            tier_only = int(sys.argv[idx + 1])

    proxy_pool = ProxyPool()
    proxy_pool.load()

    if test_mode:
        test_run(proxy_pool)
        return

    print(f"\nNBA Dataset Builder")
    print(f"{'='*60}")
    print(f"Seasons: {YEAR_START} - {YEAR_END}")
    print(f"Output:  {DATA_DIR}/")
    print(f"Workers: {MAX_WORKERS}")
    print(f"Proxies: {len(proxy_pool.proxies)}")
    print(f"{'='*60}")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        if tier_only is None or tier_only == 1:
            collect_player_stats(proxy_pool, executor)

        if tier_only is None or tier_only == 2:
            collect_team_stats(proxy_pool, executor)
            collect_standings(proxy_pool, executor)

        if tier_only is None or tier_only == 3:
            collect_rosters(proxy_pool, executor)
            collect_schedules(proxy_pool, executor)

        if tier_only is None or tier_only == 4:
            collect_draft(proxy_pool, executor)
            collect_awards(proxy_pool, executor)

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\n{'='*60}")
    print(f"COMPLETE — {minutes}m {seconds}s")
    print(f"{'='*60}")

    print("\nDataset files:")
    for f in sorted(DATA_DIR.glob("*.csv")):
        size = f.stat().st_size
        if size > 1_000_000:
            size_str = f"{size / 1_000_000:.1f}MB"
        elif size > 1_000:
            size_str = f"{size / 1_000:.0f}KB"
        else:
            size_str = f"{size}B"
        try:
            df = pd.read_csv(f)
            print(f"  {f.name:35s} {size_str:>8s}  {len(df):>6,} rows")
        except Exception:
            print(f"  {f.name:35s} {size_str:>8s}")


if __name__ == "__main__":
    main()
