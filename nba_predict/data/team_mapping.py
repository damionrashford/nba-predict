"""
Team name normalization.

Basketball Reference uses 4+ naming conventions across different pages:
  - Current-era abbreviations: BRK, CHO, NOP, OKC, MEM
  - Historical abbreviations: NJN, CHA, CHH, NOH, NOK, SEA, VAN
  - Full names: "Los Angeles Lakers", "New Jersey Nets"
  - Full names with playoff marker: "Los Angeles Lakers*"

This module maps ALL variants to 30 canonical current-era codes.
"""

# Canonical codes (current era) — the 30 NBA teams
CANONICAL_TEAMS = frozenset({
    "ATL", "BOS", "BRK", "CHI", "CHO", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
})

# Full team name → canonical abbreviation
FULL_NAME_TO_ABBR: dict[str, str] = {
    # Current names
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "LA Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
    # Historical names (relocated/rebranded franchises)
    "New Jersey Nets": "BRK",
    "Charlotte Bobcats": "CHO",
    "New Orleans Hornets": "NOP",
    "New Orleans/Oklahoma City Hornets": "NOP",
    "Seattle SuperSonics": "OKC",
    "Vancouver Grizzlies": "MEM",
}

# Historical abbreviation → canonical abbreviation
HISTORICAL_ABBR_TO_CANONICAL: dict[str, str] = {
    "NJN": "BRK",
    "CHA": "CHO",
    "CHH": "CHO",
    "NOH": "NOP",
    "NOK": "NOP",
    "SEA": "OKC",
    "VAN": "MEM",
}


def normalize_team_name(raw: str) -> str:
    """Convert any team identifier to its canonical 3-letter code.

    Handles full names, historical abbreviations, and playoff asterisks.
    Raises ValueError for unrecognized teams.
    """
    cleaned = raw.strip().rstrip("*").strip()

    # Full name lookup
    if cleaned in FULL_NAME_TO_ABBR:
        return FULL_NAME_TO_ABBR[cleaned]

    upper = cleaned.upper()

    # Historical abbreviation lookup
    if upper in HISTORICAL_ABBR_TO_CANONICAL:
        return HISTORICAL_ABBR_TO_CANONICAL[upper]

    # Already a canonical code
    if upper in CANONICAL_TEAMS:
        return upper

    raise ValueError(f"Unknown team: {raw!r}")
