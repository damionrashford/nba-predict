"""
Social media team name resolution for Reddit and Mastodon posts.

Maps subreddit names, text mentions, and hashtags to canonical
3-letter NBA team codes (matching team_mapping.py conventions).
"""

from __future__ import annotations

from nba_predict.data.team_mapping import CANONICAL_TEAMS

# ─── Reddit Subreddit → Team Code ────────────────────────────────────────────

TEAM_SUBREDDITS: dict[str, str] = {
    "atlantahawks": "ATL",
    "bostonceltics": "BOS",
    "gonets": "BRK",
    "chicagobulls": "CHI",
    "charlottehornets": "CHO",
    "clevelandcavs": "CLE",
    "mavericks": "DAL",
    "denvernuggets": "DEN",
    "detroitpistons": "DET",
    "warriors": "GSW",
    "rockets": "HOU",
    "pacers": "IND",
    "laclippers": "LAC",
    "lakers": "LAL",
    "memphisgrizzlies": "MEM",
    "heat": "MIA",
    "mkebucks": "MIL",
    "timberwolves": "MIN",
    "nolapelicans": "NOP",
    "nyknicks": "NYK",
    "thunder": "OKC",
    "orlandomagic": "ORL",
    "sixers": "PHI",
    "suns": "PHO",
    "ripcity": "POR",
    "kings": "SAC",
    "nbaspurs": "SAS",
    "torontoraptors": "TOR",
    "utahjazz": "UTA",
    "washingtonwizards": "WAS",
}

# ─── Text Mention Patterns ───────────────────────────────────────────────────
# Order: full names first (most specific), then nicknames.
# Each entry: (lowercase_pattern, team_code)

TEAM_MENTION_PATTERNS: list[tuple[str, str]] = [
    # Full city + name (most specific — checked first)
    ("atlanta hawks", "ATL"),
    ("boston celtics", "BOS"),
    ("brooklyn nets", "BRK"),
    ("chicago bulls", "CHI"),
    ("charlotte hornets", "CHO"),
    ("cleveland cavaliers", "CLE"),
    ("dallas mavericks", "DAL"),
    ("denver nuggets", "DEN"),
    ("detroit pistons", "DET"),
    ("golden state warriors", "GSW"),
    ("houston rockets", "HOU"),
    ("indiana pacers", "IND"),
    ("los angeles clippers", "LAC"),
    ("la clippers", "LAC"),
    ("los angeles lakers", "LAL"),
    ("la lakers", "LAL"),
    ("memphis grizzlies", "MEM"),
    ("miami heat", "MIA"),
    ("milwaukee bucks", "MIL"),
    ("minnesota timberwolves", "MIN"),
    ("new orleans pelicans", "NOP"),
    ("new york knicks", "NYK"),
    ("oklahoma city thunder", "OKC"),
    ("orlando magic", "ORL"),
    ("philadelphia 76ers", "PHI"),
    ("phoenix suns", "PHO"),
    ("portland trail blazers", "POR"),
    ("sacramento kings", "SAC"),
    ("san antonio spurs", "SAS"),
    ("toronto raptors", "TOR"),
    ("utah jazz", "UTA"),
    ("washington wizards", "WAS"),
    # Nicknames (unambiguous only)
    ("celtics", "BOS"),
    ("cavaliers", "CLE"),
    ("cavs", "CLE"),
    ("mavericks", "DAL"),
    ("mavs", "DAL"),
    ("nuggets", "DEN"),
    ("pistons", "DET"),
    ("warriors", "GSW"),
    ("dubs", "GSW"),
    ("rockets", "HOU"),
    ("pacers", "IND"),
    ("clippers", "LAC"),
    ("lakers", "LAL"),
    ("grizzlies", "MEM"),
    ("grizz", "MEM"),
    ("bucks", "MIL"),
    ("timberwolves", "MIN"),
    ("wolves", "MIN"),
    ("pelicans", "NOP"),
    ("pels", "NOP"),
    ("knicks", "NYK"),
    ("thunder", "OKC"),
    ("sixers", "PHI"),
    ("76ers", "PHI"),
    ("trail blazers", "POR"),
    ("blazers", "POR"),
    ("raptors", "TOR"),
    ("spurs", "SAS"),
    ("wizards", "WAS"),
]

# ─── Mastodon Hashtags per Team ──────────────────────────────────────────────

TEAM_HASHTAGS: dict[str, list[str]] = {
    "ATL": ["AtlantaHawks", "TrueToAtlanta"],
    "BOS": ["BostonCeltics", "BleedGreen"],
    "BRK": ["BrooklynNets", "NetsWorld"],
    "CHI": ["ChicagoBulls"],
    "CHO": ["CharlotteHornets", "Hornets"],
    "CLE": ["ClevelandCavaliers", "LetEmKnow"],
    "DAL": ["DallasMavericks", "Mavs"],
    "DEN": ["DenverNuggets", "MileHighBasketball"],
    "DET": ["DetroitPistons"],
    "GSW": ["Warriors", "DubNation"],
    "HOU": ["HoustonRockets"],
    "IND": ["IndianaPacers"],
    "LAC": ["LAClippers", "ClipperNation"],
    "LAL": ["LosAngelesLakers", "LakeShow"],
    "MEM": ["MemphisGrizzlies", "GrindCity"],
    "MIA": ["MiamiHeat", "HeatCulture"],
    "MIL": ["MilwaukeeBucks", "FearTheDeer"],
    "MIN": ["Timberwolves"],
    "NOP": ["Pelicans"],
    "NYK": ["NewYorkKnicks", "Knicks"],
    "OKC": ["OKCThunder", "ThunderUp"],
    "ORL": ["OrlandoMagic"],
    "PHI": ["Sixers", "TrustTheProcess"],
    "PHO": ["PhoenixSuns", "ValleyProud"],
    "POR": ["TrailBlazers", "RipCity"],
    "SAC": ["SacramentoKings", "LightTheBeam"],
    "SAS": ["SanAntonioSpurs", "GoSpursGo"],
    "TOR": ["TorontoRaptors", "WeTheNorth"],
    "UTA": ["UtahJazz"],
    "WAS": ["WashingtonWizards"],
}


# ─── Functions ───────────────────────────────────────────────────────────────


def extract_team_mentions(text: str) -> list[str]:
    """Extract all NBA team codes mentioned in a text string.

    Scans for full names, nicknames, and 3-letter abbreviations.
    Returns deduplicated list of canonical codes.
    """
    if not text:
        return []

    lower = text.lower()
    found: list[str] = []
    seen: set[str] = set()

    # Pattern matching (full names and nicknames)
    for pattern, code in TEAM_MENTION_PATTERNS:
        if pattern in lower and code not in seen:
            found.append(code)
            seen.add(code)

    # 3-letter abbreviation matching (uppercase in original text)
    for word in text.split():
        cleaned = word.strip(".,!?()[]{}:;\"'").upper()
        if cleaned in CANONICAL_TEAMS and cleaned not in seen:
            found.append(cleaned)
            seen.add(cleaned)

    return found


def subreddit_to_team(subreddit: str) -> str | None:
    """Map a subreddit name to a team code, or None if not a team sub."""
    return TEAM_SUBREDDITS.get(subreddit.lower().strip())
