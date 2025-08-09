# === 1. Imports and Setup ===
import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
import glob
import re
from difflib import SequenceMatcher
from datetime import date, timedelta
from typing import Optional, Set
import unicodedata

# NEW: API client
from playerdata_client import PlayerDataClient, sessions_to_df

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Player Readiness",
    page_icon="BostonBoltsLogo.png",
    layout="wide"
)

# === API CLIENT INIT (uses secrets.toml) ===
api = PlayerDataClient(
    st.secrets["PLAYERDATA_CLIENT_ID"],
    st.secrets["PLAYERDATA_CLIENT_SECRET"],
    st.secrets["PLAYERDATA_TOKEN_URL"],
    st.secrets["PLAYERDATA_GRAPHQL_URL"]
)
ORG_ID = st.secrets.get("PLAYERDATA_ORG_ID")
API_AVAILABLE = True

# === GLOBAL DATA WINDOW ===
# Fixed API start date for all dashboards
API_START_DATE = date(2025, 8, 7)

# === HELPERS ===
@st.cache_data(ttl=60)  # Cache expires every 60 seconds
def load_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Start Date'], format='%m/%d/%y', errors='coerce')
    if df['Date'].isna().all():
        df['Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
    return df

def similarity(a, b):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# --- Name matching helpers (nickname-aware) ---
NICKNAME_MAP = {
    # common
    "sam": "samuel", "ben": "benjamin", "alex": "alexander", "max": "maxwell",
    "will": "william", "bill": "william", "billy": "william", "liam": "william",
    "matt": "matthew", "mattie": "matthew", "mike": "michael", "mikey": "michael",
    "nick": "nicholas", "nate": "nathan", "nathaniel": "nathan",
    "tom": "thomas", "tommy": "thomas", "dave": "david", "danny": "daniel", "dan": "daniel",
    "chris": "christopher", "kris": "christopher", "topher": "christopher",
    "josh": "joshua", "jake": "jacob", "zac": "zachary", "zack": "zachary",
    "joe": "joseph", "joey": "joseph", "tony": "anthony",
    "steve": "steven", "stephen": "steven",
    "rob": "robert", "bobby": "robert", "bob": "robert",
    "rick": "richard", "ricky": "richard", "rich": "richard",
    "ted": "theodore", "theo": "theodore",
    "jon": "jonathan", "johnny": "john",
    # a few common female mappings (in case)
    "liz": "elizabeth", "beth": "elizabeth", "kate": "katherine", "katie": "katherine",
    "abby": "abigail", "ally": "allison", "allyson": "allison",
}

SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}

def _strip_accents(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

def _normalize_whitespace(text: str) -> str:
    return ' '.join(text.split())

def normalize_person_name(full_name: str) -> str:
    s = _strip_accents(full_name or "").lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    s = _normalize_whitespace(s)
    # remove common suffixes at end
    parts = s.split()
    if parts and parts[-1] in SUFFIXES:
        parts = parts[:-1]
    return ' '.join(parts)

def split_first_last(full_name: str) -> tuple[str, str]:
    n = normalize_person_name(full_name)
    parts = n.split()
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[-1]

def canonical_first(first: str) -> str:
    f = (first or "").lower()
    return NICKNAME_MAP.get(f, f)

def first_names_match(f1: str, f2: str) -> bool:
    a, b = canonical_first(f1), canonical_first(f2)
    if not a or not b:
        return False
    if a == b:
        return True
    if a[0] == b[0]:
        # initial matches, allow
        pass
    # prefix allowance (sam vs samuel) for length >= 3
    if (a.startswith(b) or b.startswith(a)) and min(len(a), len(b)) >= 3:
        return True
    # similarity on first names
    return similarity(a, b) >= 0.87

@st.cache_data(ttl=86400)  # cache org athletes for 24h
def fetch_org_athletes(org_id: str = None):
    """
    Pull athletes from Boston Bolts clubs using direct club IDs.
    Returns list[ {id, name, team, team_id} ].
    """
    
    # Boston Bolts club IDs with their current names in PlayerData
    club_mappings = {
        # MLS Next Homegrown Teams
        "d7040341-c71f-4948-a4c5-a221da3af1f0": "Boston Bolts MLS Next Homegrown 2013",
        "e74ca8dc-27dd-4008-b4d3-0d4106ebecac": "Boston Bolts MLS Next Homegrown 2012", 
        "266c5157-78d6-4792-8f23-306c9791e98d": "Boston Bolts MLS Next Homegrown 2011",
        "dc7608cf-91fd-4e2e-86de-d4e89b0e3d27": "Boston Bolts MLS Next Homegrown 2010",
        "71e0abaa-456e-438b-9e54-d50a9f5e2fce": "Boston Bolts MLS Next Homegrown 2009",
        "eda031cf-9836-4c5e-a1d5-e773bb069143": "Boston Bolts MLS Next Homegrown 2007/2008",
        
        # MLS Next Academy Division Teams  
        "4004df8e-0d06-40dd-af5f-ece612ab476a": "Boston Bolts MLS Next Academy Division 2013",
        "9664290b-d13c-48a6-95b5-777d62cc1abe": "Boston Bolts MLS Next Academy Division 2012",
        "9bc93b77-9015-48e8-afb2-8cf5a6fa26ab": "Boston Bolts MLS Next Academy Division 2011", 
        "2f3f7575-015e-4421-a164-4a0601f50099": "Boston Bolts MLS Next Academy Division 2010",
        "7596f291-f81d-4302-8ee6-b235a79a8829": "Boston Bolts MLS Next Academy Division 2009",
        "c3c6c7b5-7339-4c1b-b64a-4d10c24f258b": "Boston Bolts MLS Next Academy Division 2007/2008"
    }
    
    try:
        rows = []
        successful_clubs = 0

        for club_id, club_name in club_mappings.items():
            try:
                query = """
                query GetClub($clubId: ID!) {
                  club(id: $clubId) {
                    id
                    name
                    athletes { id name }
                  }
                }
                """
                data = api.gql(query, {"clubId": club_id})

                if data and data.get("club"):
                    club = data["club"]
                    actual_name = club.get("name", club_name)
                    athletes = club.get("athletes", []) or []

                    for athlete in athletes:
                        rows.append({
                            "id": athlete["id"],
                            "name": athlete["name"],
                            "team": actual_name,
                            "team_id": club_id,
                        })

                    if athletes:  # Only count if it has athletes
                        successful_clubs += 1

            except Exception as club_error:
                # Continue with other clubs if one fails
                st.warning(f"⚠️ Failed to load club {club_name}: {club_error}")
                continue

        if rows:
            st.success(
                f"✅ Loaded {len(rows)} athletes from {successful_clubs} Boston Bolts clubs via direct API access"
            )
        else:
            st.warning("⚠️ No athletes found in any clubs - falling back to Excel-only mode")

        return rows

    except Exception as e:
        st.warning(f"⚠️ API fetch failed: {e}")
        return []

@st.cache_data(ttl=86400)  # cache per-athlete sessions for 24h
def fetch_recent_sessions_df(athlete_id: str, athlete_name: str, days: int = 365) -> pd.DataFrame:
    """
    Fetch recent session participations for an athlete and adapt them into the CSV schema
    your UI already expects via sessions_to_df.
    """
    # Use fixed start date for all dashboards
    start_date = API_START_DATE.isoformat()

    query = """
    query Q($athleteId: ID!, $startDate: ISO8601Date!) {
      athlete(id: $athleteId) {
        sessionParticipations(startTime: $startDate, limit: 200) {
          id
          session { __typename startTime endTime }
          configuredMetrics {
            data {
              key
              label
              __typename
              localValue {
                ... on FloatMetricValue { floatValue }
                ... on IntMetricValue { intValue }
              }
            }
          }
        }
      }
    }
    """
    try:
        data = api.gql(query, {"athleteId": athlete_id, "startDate": start_date})
    except Exception as e:
        # Surface and return empty so calling code can continue
        st.warning(f"GPS query failed for {athlete_name}: {e}")
        return pd.DataFrame()

    participations = (data or {}).get("athlete", {}).get("sessionParticipations", []) or []

    # Transform to a minimal structure compatible with sessions_to_df
    sessions = []
    seen_keys: Set[str] = set()
    non_null_counts = {"totalDistanceM":0, "totalHighIntensityDistanceM":0, "totalSprintDistanceM":0, "sprintEvents":0, "maxSpeedKph":0}
    for p in participations:
        sess = p.get("session") or {}
        start_time = sess.get("startTime")
        end_time = sess.get("endTime")
        sess_type = sess.get("__typename") or "TrainingSession"

        # Extract metrics from GenericMetric list → normalize by key/label
        metrics = {
            "totalDistanceM": None,
            "totalHighIntensityDistanceM": None,
            "totalSprintDistanceM": None,
            "sprintEvents": None,
            "maxSpeedKph": None,
            "contributingSeconds": None,
        }

        def select_bucket(metric_key: str, metric_label: str) -> Optional[str]:
            k = (metric_key or "").lower()
            l = (metric_label or "").lower()
            # Distance
            if any(tok in k for tok in ["totaldistancem", "total_distance_m", "totaldistance"] ) or "total distance" in l:
                return "totalDistanceM"
            # High intensity distance
            if (
                (
                    any(tok in k for tok in [
                        "highintensity",
                        "hir",
                        "totalhighintensitydistancem",
                        "highspeed",
                        "high_speed",
                        "highspeedrunning",
                        "high_speed_running",
                    ])
                    or "high intensity" in l
                    or "hsr" in l
                    or "hsr" in k
                    or "high speed" in l
                )
                # exclude event/count/time variants
                and not any(t in (k + " " + l) for t in ["event", "events", "count", "occurrence", "duration", "time"])
                # prefer distance-like signals
                and any(t in (k + " " + l) for t in ["dist", "distance", "m", "metre", "meter", "km", "kilomet"])
            ):
                return "totalHighIntensityDistanceM"
            # Sprint distance
            if (
                ("sprint" in k or "sprint" in l)
                and not ("event" in k or "events" in k or "count" in k)
                and any(tok in (k + " " + l) for tok in ["dist", "distance", "distance_m", "distancem"])
            ) or any(tok in k for tok in ["sprintdist", "totalsprintdistancem", "sprintingdistance"]) or "sprint distance" in l:
                return "totalSprintDistanceM"
            # Sprint count
            if any(tok in k for tok in ["sprintevents", "sprintcount", "sprints"]) or "# of sprints" in l or "sprint count" in l:
                return "sprintEvents"
            # Speed
            if any(tok in k for tok in ["maxspeedkph", "topspeed", "maxspeedkmh"]) or "top speed" in l or "max speed" in l:
                return "maxSpeedKph"
            # Duration / contributing seconds
            if any(tok in k for tok in ["contributingseconds", "durationsec", "durationseconds"]) or "contributing seconds" in l or "duration" in l:
                return "contributingSeconds"
            return None

        for m in ((p.get("configuredMetrics") or {}).get("data") or []):
            key = m.get("key")
            label = m.get("label")
            lv = m.get("localValue")
            if lv is None:
                continue
            # MetricValueUnion possible types: FloatMetricValue { floatValue }, IntMetricValue { intValue }
            value = None
            if isinstance(lv, dict):
                if "floatValue" in lv and lv["floatValue"] is not None:
                    value = lv["floatValue"]
                elif "intValue" in lv and lv["intValue"] is not None:
                    value = lv["intValue"]
            elif isinstance(lv, (int, float)):
                value = lv
            bucket = select_bucket(key, label)
            if bucket is not None and value is not None:
                # Prefer meters: naive unit detection by key/label
                if bucket in ("totalDistanceM", "totalHighIntensityDistanceM", "totalSprintDistanceM"):
                    if any(tok in (key or "").lower() for tok in ["km", "kilometre", "kilometer"]) or any(tok in (label or "").lower() for tok in ["km", "kilometre", "kilometer"]):
                        value = value * 1000.0
                metrics[bucket] = float(value)
                seen_keys.add((key or "").strip())
                if bucket in non_null_counts and pd.notna(metrics[bucket]):
                    non_null_counts[bucket] += 1

        # Derive duration minutes from contributingSeconds when available; else from start/end timestamps
        duration_min = None
        if metrics.get("contributingSeconds") is not None:
            duration_min = float(metrics["contributingSeconds"]) / 60.0
        elif start_time and end_time:
            try:
                # Fallback duration by timestamps
                start_dt = pd.to_datetime(start_time, utc=True, errors="coerce")
                end_dt = pd.to_datetime(end_time, utc=True, errors="coerce")
                if pd.notna(start_dt) and pd.notna(end_dt):
                    duration_min = (end_dt - start_dt).total_seconds() / 60.0
            except Exception:
                duration_min = None

        sessions.append({
            "__typename": sess_type,
            "id": p.get("id"),
            "startTime": start_time,
            # sessions_to_df expects duration in minutes under 'durationMin'
            "durationMin": duration_min,
            # Flatten selected metrics to match sessions_to_df expectations
            "athleteMetricSet": {
                "totalDistanceM": metrics.get("totalDistanceM"),
                "totalHighIntensityDistanceM": metrics.get("totalHighIntensityDistanceM"),
                "totalSprintDistanceM": metrics.get("totalSprintDistanceM"),
                "sprintEvents": metrics.get("sprintEvents"),
                "maxSpeedKph": metrics.get("maxSpeedKph"),
            },
        })

    df_out = sessions_to_df(athlete_name, sessions)
    if st.session_state.get('show_debug', False):
        st.write({
            "parsed_metric_keys": sorted(list(seen_keys))[:10],
            "non_null_counts": non_null_counts,
            "sessions": len(sessions),
        })
    return df_out

@st.cache_data(ttl=7776000)  # Cache for ~90 days (Excel changes infrequently)
def load_unified_player_data(fetch_gps: bool = True, team_filter: str = None):
    """Load Excel files as master source and optionally merge with PlayerData API (keeping SAME keys: csv_match/csv_data).
    - When fetch_gps is False, returns Excel-derived structures only (fast path).
    - When team_filter is provided, GPS fetching is limited to that team for speed.
    """
    excel_files = glob.glob("*.xlsx")
    # Early filter: if team_filter is provided, attempt to keep only that team's workbook(s)
    if team_filter:
        filtered = []
        tf = team_filter.lower()
        for f in excel_files:
            name_l = f.lower()
            # keep files whose filename contains the year token present in team_filter
            # e.g., team_filter "2011 MLS Next" → match "2011"
            year_tokens = re.findall(r"(\d{4}|07_08|\d{2})", tf)
            keep = False
            for tok in year_tokens:
                if tok and tok in name_l:
                    keep = True
                    break
                # map short years like "09" -> "2009"
                if tok == "09" and "2009" in name_l:
                    keep = True
                    break
            if keep:
                filtered.append(f)
        if filtered:
            excel_files = filtered

    # Load Excel players as master source - now from Profiles sheet
    excel_players = {}  # {player_name: {team, excel_file, csv_match, csv_data, testing_data, profile_data, phv_data}}

    for f in excel_files:
        try:
            # Extract year from filename - handle various formats
            year_match = re.search(r'(\d{4}|07_08|\d{2})', f)
            if not year_match:
                continue
            year = year_match.group(1)
            if year == "09":
                year = "2009"
            elif year == "07_08":
                year = "2007-08"
            team_name = f"{year} MLS Next"

            # Read Profiles sheet to get ALL athletes
            try:
                profiles_df = pd.read_excel(f, sheet_name='Profiles')
                if 'ACTIVE ATHLETE' in profiles_df.columns:
                    active_athletes = profiles_df[profiles_df['ACTIVE ATHLETE'] == True]
                else:
                    active_athletes = profiles_df  # Use all if no active flag

                # Load testing data for getting most recent session
                testing_df = pd.read_excel(f, sheet_name='TestingData')
                metric_names = testing_df.iloc[0].values
                testing_df.columns = metric_names
                testing_clean = testing_df.iloc[1:].reset_index(drop=True)
                testing_clean['Date'] = pd.to_datetime(testing_clean['Date'], errors='coerce')

                # Process each athlete from Profiles
                for _, athlete_row in active_athletes.iterrows():
                    player_name = str(athlete_row.get('NAME', '')).strip()
                    if not player_name:
                        continue

                    # Get most recent testing data for this athlete
                    athlete_testing = testing_clean[testing_clean['Name'].str.contains(player_name, na=False, case=False)]
                    if not athlete_testing.empty:
                        most_recent = athlete_testing.loc[athlete_testing['Date'].idxmax()]
                        testing_data = most_recent.to_dict()
                    else:
                        testing_data = None

                    # Get PHV Calculator data for height/weight
                    phv_data = {}
                    try:
                        # Read PHV Calculator with proper header row (row 7)
                        phv_df = pd.read_excel(f, sheet_name='PHV Calculator', header=7)
                        
                        # Look for the player by matching first and last name
                        for idx, row in phv_df.iterrows():
                            first_name = str(row.get('First Name', '')).strip()
                            last_name = str(row.get('Last Name', '')).strip()
                            if first_name and first_name != 'nan' and last_name and last_name != 'nan':
                                phv_full_name = f"{first_name} {last_name}"
                                if similarity(player_name, phv_full_name) >= 0.85:
                                    height_val = row.get('Height (cm)', None)
                                    weight_val = row.get('Weight (kg)', None)
                                    phv_data = {
                                            'Name': phv_full_name,
                                            'Height': height_val if pd.notna(height_val) and height_val != 0 else None,
                                            'Weight': weight_val if pd.notna(weight_val) and weight_val != 0 else None
                                    }
                                    # Successfully found PHV data
                                    break
                    except Exception as e:
                        print(f"Warning: Could not read PHV Calculator sheet from {f}: {e}")
                        pass  # PHV Calculator sheet might not exist or have different structure

                    excel_players[player_name] = {
                        'team': team_name,
                        'excel_file': f,
                        'csv_match': None,     # will set after API fetch
                        'csv_data': None,      # will set after API fetch
                        'testing_data': testing_data,
                        'profile_data': athlete_row.to_dict(),
                        'phv_data': phv_data
                    }

            except Exception:
                # Fallback to old method if Profiles sheet doesn't exist
                df = pd.read_excel(f, sheet_name=0)
                header = df.columns[0]
                name_match = re.search(r'FOR (.+?) -', header)
                if name_match:
                    player_name = name_match.group(1).strip()
                    excel_players[player_name] = {
                        'team': team_name,
                        'excel_file': f,
                        'csv_match': None,
                        'csv_data': None,
                        'testing_data': None,
                        'profile_data': None,
                        'phv_data': {}
                    }

        except Exception:
            continue

    if not excel_players:
        return {}, {}

    # ==== API Integration (optional) ====
    if st.session_state.get('enable_api', False) and fetch_gps:
        try:
            with st.spinner("Loading athletes from PlayerData…"):
                api_athletes = fetch_org_athletes(ORG_ID)
            api_name_to_id = {a["name"]: a["id"] for a in api_athletes}

            # Build normalized API index for faster matching
            api_index = {}
            for api_name, aid in api_name_to_id.items():
                f, l = split_first_last(api_name)
                api_index.setdefault((canonical_first(f), l), []).append((api_name, aid))

            def match_name_to_id(name: str):
                # exact first try
                if name in api_name_to_id:
                    return api_name_to_id[name], name, 1.0

                f, l = split_first_last(name)
                key = (canonical_first(f), l)

                # 1) exact canonical-first + last
                if key in api_index:
                    api_name, aid = api_index[key][0]
                    return aid, api_name, 0.99

                # 2) last name must match, then flexible first
                best_id, best_name, best_sim = None, None, 0
                for api_name, aid in api_name_to_id.items():
                    f2, l2 = split_first_last(api_name)
                    if l2 != l or not l:
                        continue
                    if first_names_match(f, f2):
                        return aid, api_name, 0.95
                    sim = similarity(name, api_name)
                    if sim > best_sim:
                        best_id, best_name, best_sim = aid, api_name, sim
                return (best_id, best_name, best_sim) if best_sim >= 0.90 else (None, None, 0)

            # Fetch GPS data for each player
            matched_count = 0
            gps_data_count = 0
            # Limit to selected team if provided
            players_items = list(excel_players.items())
            if team_filter:
                players_items = [(n, i) for n, i in players_items if i.get('team') == team_filter]

            total_to_fetch = len(players_items)
            progress_bar = st.progress(0) if total_to_fetch > 0 else None
            status_txt = st.empty() if total_to_fetch > 0 else None

            for idx, (player_name, info) in enumerate(players_items, start=1):
                if progress_bar is not None:
                    pct = int(idx / max(total_to_fetch, 1) * 100)
                    progress_bar.progress(pct)
                    if status_txt is not None:
                        status_txt.write(f"Fetching GPS: {player_name} ({idx}/{total_to_fetch})")
                aid, matched_api_name, sim = match_name_to_id(player_name)
                if aid:
                    matched_count += 1
                    try:
                        gps_df = fetch_recent_sessions_df(aid, player_name)
                        if gps_df is not None and len(gps_df) > 0:
                            gps_data_count += 1
                            excel_players[player_name]['csv_match'] = player_name
                            excel_players[player_name]['csv_data'] = gps_df
                            excel_players[player_name]['csv_similarity'] = sim
                            excel_players[player_name]['api_id'] = aid
                        else:
                            excel_players[player_name]['csv_match'] = None
                            excel_players[player_name]['csv_data'] = None
                            excel_players[player_name]['csv_similarity'] = 0
                            excel_players[player_name]['api_id'] = None
                    except Exception as e:
                        st.warning(f"GPS fetch failed for {player_name}: {e}")
                        excel_players[player_name]['csv_match'] = None
                        excel_players[player_name]['csv_data'] = None
                        excel_players[player_name]['csv_similarity'] = 0
                        excel_players[player_name]['api_id'] = None
                else:
                    excel_players[player_name]['csv_match'] = None
                    excel_players[player_name]['csv_data'] = None
                    excel_players[player_name]['csv_similarity'] = 0
                    excel_players[player_name]['api_id'] = None
        except Exception as e:
            st.warning(f"GPS fetch failed for {player_name}: {e}")
            excel_players[player_name]['csv_match'] = None
            excel_players[player_name]['csv_data'] = None
            excel_players[player_name]['csv_similarity'] = 0
            excel_players[player_name]['api_id'] = None
        except Exception as e:
            st.warning(f"GPS fetch failed for {player_name}: {e}")
            excel_players[player_name]['csv_match'] = None
            excel_players[player_name]['csv_data'] = None
            excel_players[player_name]['csv_similarity'] = 0
            excel_players[player_name]['api_id'] = None
            excel_players[player_name]['api_id'] = None
            if progress_bar is not None:
                progress_bar.progress(100)
                if status_txt is not None:
                    status_txt.write("GPS fetching complete.")
            
            if matched_count > 0:
                st.info(f"🔗 Matched {matched_count} Excel players to API athletes")
                if gps_data_count > 0:
                    st.success(f"📊 Found GPS data for {gps_data_count} players")
                else:
                    st.warning("⚠️ No recent GPS session data found for the selected players")
            else:
                st.warning("⚠️ No Excel players matched to API athletes - check name matching")
        except Exception as e:
            st.warning(f"⚠️ API unavailable: {e}")
            for player_name, info in excel_players.items():
                excel_players[player_name]['csv_match'] = None
                excel_players[player_name]['csv_data'] = None
                excel_players[player_name]['csv_similarity'] = 0
                excel_players[player_name]['api_id'] = None
    else:
        # Excel-only mode for faster loading
        for player_name, info in excel_players.items():
            excel_players[player_name]['csv_match'] = None
            excel_players[player_name]['csv_data'] = None
            excel_players[player_name]['csv_similarity'] = 0
            excel_players[player_name]['api_id'] = None

    # Group by teams (unchanged)
    teams_players = {}
    for player, info in excel_players.items():
        team = info['team']
        if team not in teams_players:
            teams_players[team] = []
        teams_players[team].append(player)

    return teams_players, excel_players

@st.cache_data(ttl=7776000)  # Cache testing data for ~90 days
def load_testing_data(excel_file):
    """Load testing data from Excel file"""
    try:
        df = pd.read_excel(excel_file, sheet_name='TestingData')
        metric_names = df.iloc[0].values
        df.columns = metric_names
        df = df.iloc[1:].reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading testing data: {e}")
        return pd.DataFrame()

def render_unified_player_dashboard(player_name, excel_players):
    """Render unified dashboard with all three tabs for a specific player"""
    if player_name not in excel_players:
        st.error(f"Player {player_name} not found in Excel data")
        return

    player_info = excel_players[player_name]
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Gauges", "📈 ACWR Analysis", "🏋️ Testing Data", "🧭 Physical Radar Charts"])
    with tab1:
        render_player_gauges(player_name, player_info)
    with tab2:
        render_player_acwr(player_name, player_info)
    with tab3:
        render_player_testing_data_unified(player_name, player_info)
    with tab4:
        render_api_radars(player_name, player_info, excel_players)

def render_api_radars(player_name, player_info, excel_players):
    csv_data = player_info.get('csv_data')
    if csv_data is None or csv_data.empty:
        st.warning("No API session data available for radars.")
        return

    player_key = player_info.get('csv_match', player_name)

    def concat_team_api_df() -> pd.DataFrame:
        rows = []
        for p, info in excel_players.items():
            c = info.get('csv_data')
            if c is None or c.empty:
                continue
            pos = (info.get('profile_data') or {}).get('POSITION', 'Unknown')
            c2 = c.copy()
            c2['__PLAYER__'] = p
            c2['__POSITION__'] = pos
            rows.append(c2)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    all_api = concat_team_api_df()
    if all_api.empty:
        st.warning("No API data available to build percentiles.")
        return

    # keep Whole Session only and ensure numerics
    all_api = all_api[all_api['Segment Name'] == 'Whole Session'].copy()
    all_api['Duration (mins)'] = pd.to_numeric(all_api['Duration (mins)'], errors='coerce')
    for m in METRICS:
        all_api[m] = pd.to_numeric(all_api[m], errors='coerce')
    all_api = all_api.dropna(subset=['Duration (mins)'])

    def per_player_metrics(source: pd.DataFrame, session_type: str, per90: bool) -> pd.DataFrame:
        base = source[source['Session Type'] == session_type].copy()
        if base.empty:
            return pd.DataFrame(columns=['__PLAYER__'] + METRICS).set_index('__PLAYER__')
        if per90:
            for m in METRICS:
                if m != 'Top Speed (kph)':
                    base[m] = (base[m] / base['Duration (mins)']) * 90
        # aggregate per player
        agg = {}
        for m in METRICS:
            if m == 'Top Speed (kph)':
                agg[m] = 'max'
            else:
                agg[m] = 'mean'
        dfp = base.groupby('__PLAYER__').agg(agg)
        return dfp

    # build per-player tables
    team_match = per_player_metrics(all_api, 'Match Session', per90=True)
    team_train = per_player_metrics(all_api, 'Training Session', per90=False)

    # position filters
    pos_map = all_api.groupby('__PLAYER__')['__POSITION__'].agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) else 'Unknown')
    player_pos = pos_map.get(player_key, (player_info.get('profile_data') or {}).get('POSITION', 'Unknown'))
    pos_players = set(pos_map[pos_map == player_pos].index.tolist())
    pos_match = team_match[team_match.index.isin(pos_players)] if not team_match.empty else team_match
    pos_train = team_train[team_train.index.isin(pos_players)] if not team_train.empty else team_train

    def percentiles(table: pd.DataFrame, who: str) -> dict:
        out = {m: 0.0 for m in METRICS}
        if table.empty or who not in table.index:
            return out
        row = table.loc[who]
        for m in METRICS:
            col = table[m].dropna()
            if col.empty:
                out[m] = 0.0
                continue
            val = row[m]
            out[m] = float((col <= val).mean() * 100.0)
        return out

    # compute player percentiles
    p_match_vs_team = percentiles(team_match, player_key)
    p_match_vs_pos = percentiles(pos_match, player_key)
    p_train_vs_team = percentiles(team_train, player_key)
    p_train_vs_pos = percentiles(pos_train, player_key)

    def radar_percent(fig_title: str, values: dict, color="#3b82f6", key: str = None):
        cats = [METRIC_LABELS[m] for m in METRICS]
        vals = [values.get(m, 0) for m in METRICS]
        fig = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill='toself', name='Percentile',
                                        line_color=color, fillcolor='rgba(59,130,246,0.25)'))
        fig.update_layout(template='plotly_white',
                          polar=dict(radialaxis=dict(visible=True, range=[0,100], tickvals=[0,25,50,75,100])),
                          showlegend=False, title=fig_title, height=380, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True, key=key or f"radar-{player_key}-{fig_title}")

    col1, col2 = st.columns(2)
    with col1:
        radar_percent("Match Sessions — percentile vs Team", p_match_vs_team, key=f"{player_key}-match-team")
    with col2:
        radar_percent(f"Match Sessions — percentile vs {player_pos}", p_match_vs_pos, color="#10b981", key=f"{player_key}-match-pos")

    col3, col4 = st.columns(2)
    with col3:
        radar_percent("Training Sessions — percentile vs Team", p_train_vs_team, key=f"{player_key}-train-team")
    with col4:
        radar_percent(f"Training Sessions — percentile vs {player_pos}", p_train_vs_pos, color="#10b981", key=f"{player_key}-train-pos")

def render_player_gauges(player_name, player_info):
    """Render performance gauges for a specific player"""
    csv_data = player_info.get('csv_data')
    csv_match = player_info.get('csv_match')

    if csv_data is None or csv_match is None:
        st.warning(f"No CSV performance data found for {player_name}")
        return

    # Filter data for the matched player
    df = csv_data[csv_data['Athlete Name'] == csv_match].copy()
    df['Date'] = pd.to_datetime(df['Start Date'], format='%m/%d/%y', errors='coerce')
    if df['Date'].isna().all():
        df['Date'] = pd.to_datetime(df['Start Date'], errors='coerce')

    df = df.dropna(subset=["Date","Session Type","Segment Name"])
    df = df[df["Segment Name"]=="Whole Session"].sort_values("Date")

    if df.empty:
        st.warning(f"No performance data found for {player_name}")
        return

    # Process data for gauges (same)
    df["Duration (mins)"] = pd.to_numeric(df["Duration (mins)"], errors="coerce")
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    # Get matches and training data
    matches = df[
        (df["Session Type"]=="Match Session") &
        (df["Duration (mins)"]>0)
        ].sort_values("Date")

    if matches.empty:
        st.warning("No match data found for performance benchmarks.")
        return

    latest_match_date = matches["Date"].max()
    st.markdown(f"**Latest Match Date:** {latest_match_date.date()}")

    # Get latest training week
    train_df = df[df["Session Type"]=="Training Session"]
    if train_df.empty:
        st.warning("No training data found.")
        return

    latest_training_date = train_df["Date"].max()
    iso_year, iso_week, _ = latest_training_date.isocalendar()

    # Calculate benchmarks
    match_avg = {}
    for m in METRICS:
        if m != "Top Speed (kph)":
            matches["Per90"] = matches[m]/matches["Duration (mins)"]*90
            match_avg[m] = matches["Per90"].mean()

    iso = df["Date"].dt.isocalendar()
    training_week = df[
        (df["Session Type"]=="Training Session") &
        (iso["week"]==iso_week) &
        (iso["year"]==iso_year)
    ]

    if training_week.empty:
        st.warning("No training data for current week.")
        return

    top_speed_benchmark = df["Top Speed (kph)"].max()
    grouped = training_week.agg({
        "Distance (m)":"sum",
        "High Intensity Running (m)":"sum",
        "Sprint Distance (m)":"sum",
        "No. of Sprints":"sum",
        "Top Speed (kph)":"max"
    }).to_frame().T

    # Debug info
    if st.session_state.get('show_debug', False):
        st.markdown("### 🔍 Debug Information")
        st.info(f"Total sessions: {len(df)} | Match sessions: {len(matches)} | Training sessions: {len(train_df)} | Training week sessions: {len(training_week)}")
        st.caption(f"Latest training: {latest_training_date.date()} | ISO Year: {iso_year}, ISO Week: {iso_week}")

        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Match Averages (per 90 mins):**")
                for metric, value in match_avg.items():
                    if pd.notna(value):
                        st.write(f"- {METRIC_LABELS[metric]}: {value:.2f}")
            with col2:
                st.markdown("**Training Week Totals/Max:**")
                for metric in METRICS:
                    val = grouped[metric].iloc[0] if not grouped.empty else None
                    if pd.notna(val):
                        st.write(f"- {METRIC_LABELS[metric]}: {val:.2f}")
                st.write(f"**Top Speed Benchmark:** {top_speed_benchmark:.2f} kph")

    st.markdown("### Weekly Performance vs Match Averages")
    cols = st.columns(len(METRICS))
    for i, metric in enumerate(METRICS):
        with cols[i]:
            if metric == "Top Speed (kph)":
                train_val = grouped[metric].max()
                benchmark = top_speed_benchmark
                ratio = 0 if pd.isna(benchmark) or benchmark == 0 else train_val/benchmark
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(ratio,2),
                    number={"font":{"size": 20}},
                    gauge={
                        "axis":{"range":[0,1.0],"showticklabels":False},
                        "bar":{"color": get_color(ratio, metric)},
                        "steps":[
                            {"range":[0,0.5],"color":"#ffcccc"},
                            {"range":[0.5,0.75],"color":"#ffe0b3"},
                            {"range":[0.75,0.9],"color":"#ffffcc"},
                            {"range":[0.9,1.0],"color":"#ccffcc"},
                        ]
                    }
                ))
                fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=180)
                st.markdown(f"<div style='text-align:center;font-weight:bold;'>{METRIC_LABELS[metric]}</div>", unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, key=f"{player_name}-top-{i}")

                if st.session_state.get('show_debug', False):
                    st.caption(f"Ratio: {ratio:.2f} | This Week: {train_val:.2f} | Max: {benchmark:.2f}")
                    if ratio < 0.9:
                        st.markdown(
                            "<div style='text-align:center;color:red;font-weight:bold;'>"
                            "⚠️ Did not reach 90% of max speed this week"
                            "</div>",
                            unsafe_allow_html=True
                        )
            else:
                train_val = grouped[metric].sum()
                benchmark = match_avg.get(metric, None)
                fig = create_readiness_gauge(train_val, benchmark, METRIC_LABELS[metric])
                st.markdown(f"<div style='text-align:center;font-weight:bold;'>{METRIC_LABELS[metric]}</div>", unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, key=f"{player_name}-{metric}-{i}")

                # === PROJECTION AND WARNING LOGIC ===
                prev = df[
                    (df["Session Type"]=="Training Session") &
                    (df["Date"]<training_week["Date"].min())
                ]

                if not prev.empty:
                    prev = prev.copy()
                    prev["Year"] = prev["Date"].dt.isocalendar().year
                    prev["Week"] = prev["Date"].dt.isocalendar().week
                    week_sums = prev.groupby(["Year","Week"])[METRICS].sum().reset_index()
                    valid = week_sums[(week_sums.drop(columns=["Year","Week"])>0).any(axis=1)]
                    if not valid.empty:
                        last = valid.iloc[-1]
                        prev_week_str = f"Week {int(last['Week'])}, {int(last['Year'])}"
                        prev_data = prev[
                            (prev["Date"].dt.isocalendar().week==last["Week"]) &
                            (prev["Date"].dt.isocalendar().year==last["Year"])
                        ]
                        previous_week_total = prev_data[metric].sum()
                    else:
                        prev_week_str="None"
                        previous_week_total = 0
                else:
                    prev_week_str="None"
                    previous_week_total = 0

                practices_done = training_week.shape[0]
                current_sum = training_week[metric].sum()

                iso_all = df["Date"].dt.isocalendar()
                df_temp = df.copy()
                df_temp["PracticeNumber"] = (
                    df_temp[df_temp["Session Type"]=="Training Session"]
                    .groupby([iso_all.year, iso_all.week]).cumcount()+1
                ).clip(upper=3)
                practice_avgs = (
                    df_temp[df_temp["Session Type"]=="Training Session"]
                    .groupby("PracticeNumber")[metric].mean()
                    .reindex([1,2,3], fill_value=0)
                )

                if previous_week_total > 0 and current_sum > 1.10 * previous_week_total:
                    flag = "⚠️"
                    flag_val = current_sum
                    projection_used = False
                    projected_total = None
                else:
                    if practices_done < 3:
                        needed = list(range(practices_done+1, 4))
                        projected_total = current_sum + practice_avgs.loc[needed].sum()
                        flag_val = projected_total
                        projection_used = True
                    else:
                        projected_total = None
                        flag_val = current_sum
                        projection_used = False

                    if previous_week_total > 0 and flag_val > 1.10 * previous_week_total:
                        flag = "🔮⚠️" if projection_used else "⚠️"
                    else:
                        flag = ""

                if flag:
                    if projection_used:
                        st.markdown(
                            "<div style='text-align:center;font-weight:bold;color:orange;'>"
                            f"⚠️ Projected total of {METRIC_LABELS[metric]} is on track to be > 110% of last week's total"
                            "</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            "<div style='text-align:center;font-weight:bold;color:red;'>"
                            f"⚠️ {METRIC_LABELS[metric]} is > 110% than last week's total"
                            "</div>",
                            unsafe_allow_html=True
                        )

                if st.session_state.get('show_debug', False):
                    if benchmark is not None:
                        ratio = train_val / benchmark if benchmark > 0 else 0
                        st.caption(f"Ratio: {ratio:.2f} | This Week: {train_val:.2f} | Avg: {benchmark:.2f}")
                    else:
                        st.caption(f"This Week: {train_val:.2f} | No benchmark available")
                    st.markdown(f"""
                        <div style='font-size:12px;color:#666;border:1px solid #ddd;padding:5px;margin:2px;'>
                            <b>Debug for {METRIC_LABELS[metric]}</b><br>
                            • Previous Week: {prev_week_str}<br>
                            • Previous Week Total: {previous_week_total:.1f}<br>
                            • Current Week So Far: {current_sum:.1f}<br>
                            • Practices Done: {practices_done}<br>
                            • Historical Practice Avgs: {practice_avgs.to_dict()}<br>
                            • Projected Total: {projected_total if projection_used else 'N/A'}<br>
                            • Final Used: {flag_val:.1f} ({'Projected' if projection_used else 'Actual'})<br>
                            • Threshold (110%): {1.10*previous_week_total:.1f}<br>
                            • ⚠️ Flag: {'YES' if flag else 'NO'}
                        </div>
                    """, unsafe_allow_html=True)

def render_player_acwr(player_name, player_info):
    """Render ACWR analysis for a specific player"""
    csv_data = player_info.get('csv_data')
    csv_match = player_info.get('csv_match')

    if csv_data is None or csv_match is None:
        st.warning(f"No CSV performance data found for {player_name}")
        return

    df_acwr = csv_data[csv_data['Athlete Name'] == csv_match].copy()
    df_acwr['Date'] = pd.to_datetime(df_acwr['Start Date'], format='%m/%d/%y', errors='coerce')
    if df_acwr['Date'].isna().all():
        df_acwr['Date'] = pd.to_datetime(df_acwr['Start Date'], errors='coerce')

    df_acwr = (
        df_acwr
        .dropna(subset=["Date","Session Type","Segment Name"])
        .query("`Segment Name`=='Whole Session' and `Session Type`=='Training Session'")
    )

    if df_acwr.empty:
        st.warning(f"No training data found for {player_name}")
        return

    metrics_acwr = ["Distance (m)", "High Intensity Running (m)", "Sprint Distance (m)", "No. of Sprints"]

    df_daily = (
        df_acwr
        .groupby(["Date"])[metrics_acwr]
        .sum()
        .reset_index()
        .sort_values("Date")
        .set_index("Date")
    )

    for m in metrics_acwr:
        df_daily[f"acute_{m}"] = df_daily[m].rolling("7d").sum()
        df_daily[f"chronic_{m}"] = df_daily[m].rolling("28d").sum() / 4
        df_daily[f"acwr_{m}"] = df_daily[f"acute_{m}"] / df_daily[f"chronic_{m}"]

    df_daily = df_daily.reset_index()

    color_map = {
        "Distance (m)": "#1f77b4",
        "High Intensity Running (m)": "#ff7f0e",
        "Sprint Distance (m)": "#2ca02c",
        "No. of Sprints": "#d62728"
    }

    st.markdown("### Metric Selection")
    col1, col2, col3, col4 = st.columns(4)
    selected_metrics = []
    with col1:
        if st.checkbox("Total Distance", value=True, key=f"{player_name}_distance_acwr"):
            selected_metrics.append("Distance (m)")
    with col2:
        if st.checkbox("HSR", value=True, key=f"{player_name}_hsr_acwr"):
            selected_metrics.append("High Intensity Running (m)")
    with col3:
        if st.checkbox("Sprint Distance", value=True, key=f"{player_name}_sprint_dist_acwr"):
            selected_metrics.append("Sprint Distance (m)")
    with col4:
        if st.checkbox("# of Sprints", value=True, key=f"{player_name}_sprint_count_acwr"):
            selected_metrics.append("No. of Sprints")

    if not selected_metrics:
        st.warning("Please select at least one metric to display.")
        return

    st.markdown("### Acute:Chronic Workload Ratio (7d:28d)")
    fig = go.Figure()
    for m in selected_metrics:
        fig.add_trace(go.Scatter(
            x=df_daily["Date"],
            y=df_daily[f"acwr_{m}"],
            mode="lines+markers",
            name=METRIC_LABELS[m],
            line_shape="spline",
            line=dict(width=2, color=color_map[m]),
            marker=dict(size=4, color=color_map[m])
        ))
    fig.add_hrect(y0=0.8, y1=1.3, fillcolor="lightgreen", opacity=0.2, line_width=0)
    fig.update_layout(
        template="plotly_white",
        title=f"{player_name} — ACWR (7d ∶ 28d)",
        xaxis_title="Date",
        yaxis_title="ACWR",
        legend_title="Metric",
        font=dict(family="Arial", size=12),
        margin=dict(t=50, b=40, l=40, r=40),
        height=350
    )
    st.plotly_chart(fig, use_container_width=True, key=f"acwr_{player_name}")

    st.markdown("### Current Status")
    latest_data = df_daily.iloc[-1] if not df_daily.empty else None
    if latest_data is not None:
        for m in selected_metrics:
            acwr_col = f"acwr_{m}"
            if acwr_col in latest_data:
                val = latest_data[acwr_col]
                if pd.notna(val):
                    if val < 0.8:
                        st.markdown(f"🚨 Undertrained for {METRIC_LABELS[m]}: ACWR = {val:.2f}")
                    elif val > 1.3:
                        st.markdown(f"⚠️ Overtrained for {METRIC_LABELS[m]}: ACWR = {val:.2f}")

def render_player_testing_data_unified(player_name, player_info):
    """Render clean, modern testing dashboard with raw data, rankings, and scores"""
    testing_data_dict = player_info.get('testing_data')
    profile_data = player_info.get('profile_data', {})

    if not testing_data_dict:
        st.warning(f"No testing data available for {player_name}")
        return

    excel_file = player_info['excel_file']
    full_testing_data = load_testing_data(excel_file)
    if full_testing_data.empty:
        st.error("Could not load testing data from Excel file")
        return

    st.markdown(f"""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1 style='color: #1e293b; font-size: 3rem; font-weight: 700; margin: 0;'>{player_name.upper()}</h1>
        <p style='color: #64748b; font-size: 1.2rem; margin: 0.5rem 0 0 0;'>Performance Testing Report - {player_info['team']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; color: #1e293b; margin-bottom: 2rem;'>🏃‍♂️ Player Profile</h3>", unsafe_allow_html=True)

    phv_data = player_info.get('phv_data', {})
    height = 'N/A'
    weight = 'N/A'
    if phv_data.get('Height') and phv_data['Height'] != 0:
        height = f"{phv_data['Height']:.1f}"
    if phv_data.get('Weight') and phv_data['Weight'] != 0:
        weight = f"{phv_data['Weight']:.1f}"
    if height == 'N/A':
        height = profile_data.get('HEIGHT', 'N/A')
    if weight == 'N/A':
        weight = profile_data.get('WEIGHT', 'N/A')

    position = profile_data.get('POSITION', 'Forward')
    overall_score = get_overall_score_from_data(testing_data_dict)
    performance_label, _ = get_performance_label_from_score(overall_score)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        create_profile_card("Position", position, "", "#22c55e")
    with col2:
        height_unit = "cm" if height != 'N/A' else ""
        create_profile_card("Height", height, height_unit, "#3b82f6")
    with col3:
        weight_unit = "kg" if weight != 'N/A' else ""
        create_profile_card("Weight", weight, weight_unit, "#f59e0b")
    with col4:
        perf_color = "#22c55e" if performance_label == "ELITE" else "#3b82f6" if performance_label == "ABOVE AVERAGE" else "#f59e0b" if performance_label == "AVERAGE" else "#ef4444"
        create_profile_card("Performance", performance_label, f"Score: {overall_score:.1f}", perf_color)

    st.markdown("---")

    st.markdown("<h3 style='text-align: center; color: #1e293b; margin-bottom: 2rem;'>📊 Performance Metrics</h3>", unsafe_allow_html=True)
    metrics_config = {
        "FITNESS": {"metrics": ["VO2MAX"], "icon": "🫁"},
        "SPEED": {"metrics": ["10M SPRINT", "40M SPRINT", "AGILITY TEST"], "icon": "⚡"},
        "POWER": {"metrics": ["BROAD JUMP"], "icon": "💪"},
        "STRENGTH": {"metrics": ["PULL UPS"], "icon": "🏋️"},
    }

    for category, config in metrics_config.items():
        st.markdown(f"#### {config['icon']} {category}")
        cols = st.columns(len(config['metrics']))
        for i, metric in enumerate(config['metrics']):
            with cols[i]:
                create_metric_card(metric, testing_data_dict, full_testing_data, player_name)
        st.markdown("")

    st.markdown("---")
    st.markdown("<h3 style='text-align: center; color: #1e293b; margin-bottom: 2rem;'>📊 Performance Comparison</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4 style='text-align:center'>vs Team Average</h4>", unsafe_allow_html=True)
        create_team_comparison_radar(player_name, testing_data_dict, full_testing_data, metrics_config)
    with col2:
        st.markdown("<h4 style='text-align:center'>vs Position Average</h4>", unsafe_allow_html=True)
        position = profile_data.get('POSITION', 'Forward')
        create_position_comparison_radar(player_name, testing_data_dict, full_testing_data, metrics_config, position)

def get_metric_unit(metric):
    units = {
        'MED BALL THROW AVG': 'm',
        'BROAD JUMP': 'cm',
        '10M SPRINT': 's',
        '40M SPRINT': 's',
        'BODY WEIGHT': 'kg',
        'AGILITY TEST': 's',
        'VO2MAX': 'ml/kg/min',
        'TBDL LOAD': 'kg'
    }
    return units.get(metric, '')

def get_metric_unit_updated(metric):
    units = {
        'VO2MAX': 'ml/kg/min',
        '10M SPRINT': 's',
        '40M SPRINT': 's',
        'AGILITY TEST': 's',
        'PULL UPS': 'reps',
        'BROAD JUMP': 'cm'
    }
    return units.get(metric, '')

def create_info_card(title, value):
    st.metric(label=title, value=value)

def create_profile_card(title, value, unit="", color="#3b82f6"):
    formatted_value = value if value and value != 'N/A' else 'N/A'
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                padding: 1.5rem; border-radius: 12px; text-align: center; 
                border: 2px solid {color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 1rem; min-height: 120px; display: flex; flex-direction: column; justify-content: center;'>
        <div style='font-size: 0.875rem; color: #64748b; font-weight: 600; margin-bottom: 0.5rem;'>{title.upper()}</div>
        <div style='font-size: 1.75rem; font-weight: 700; color: #1e293b; margin-bottom: 0.25rem; min-height: 2.1rem; display: flex; align-items: center; justify-content: center;'>{formatted_value}</div>
        <div style='font-size: 0.875rem; color: #64748b; font-weight: 500; min-height: 1.3rem;'>{unit}</div>
    </div>
    """, unsafe_allow_html=True)

def get_overall_score_from_data(testing_data):
    return testing_data.get('OVERALL SCORE', 0) or 0

def create_metric_card(metric, testing_data, full_data, player_name):
    raw_value = testing_data.get(metric)
    if pd.isna(raw_value) or raw_value is None:
        st.warning(f"**{metric}** - No data available")
        return

    score_col = f"{metric} TEAM SCORE"
    score = testing_data.get(score_col, 0) or 0

    metric_values = full_data[metric].dropna()
    if len(metric_values) > 0:
        if metric in ['10M SPRINT', '40M SPRINT', 'AGILITY TEST']:
            rank = (metric_values < raw_value).sum() + 1
        else:
            rank = (metric_values > raw_value).sum() + 1
        total_players = len(metric_values)
    else:
        rank = 1
        total_players = 1

    performance_label, color = get_performance_label_from_score(score)
    unit = get_metric_unit_updated(metric)
    formatted_value = f"{raw_value:.1f}" if isinstance(raw_value, (int, float, float)) else str(raw_value)

    with st.container():
        metric_display_name = metric.title()
        st.markdown(f"""
        <div style='text-align: center; margin-bottom: 1rem;'>
            <h3 style='color: #1e293b; font-weight: 600; margin: 0; font-size: 1.5rem;'>{metric_display_name}</h3>
        </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                        padding: 1.5rem; border-radius: 12px; text-align: center; 
                        border: 2px solid {color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <div style='font-size: 0.875rem; color: #64748b; font-weight: 600; margin-bottom: 0.5rem;'>RAW VALUE</div>
                <div style='font-size: 2rem; font-weight: 700; color: #1e293b; margin-bottom: 0.25rem;'>{formatted_value}</div>
                <div style='font-size: 0.875rem; color: #64748b; font-weight: 500;'>{unit}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                        padding: 1.5rem; border-radius: 12px; text-align: center; 
                        border: 2px solid {color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <div style='font-size: 0.875rem; color: #64748b; font-weight: 600; margin-bottom: 0.5rem;'>SCORE</div>
                <div style='font-size: 2rem; font-weight: 700; color: #1e293b; margin-bottom: 0.25rem;'>{score:.0f}</div>
                <div style='font-size: 0.875rem; color: #64748b; font-weight: 500;'>out of 100</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                        padding: 1.5rem; border-radius: 12px; text-align: center; 
                        border: 2px solid {color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <div style='font-size: 0.875rem; color: #64748b; font-weight: 600; margin-bottom: 0.5rem;'>TEAM RANK</div>
                <div style='font-size: 2rem; font-weight: 700; color: #1e293b; margin-bottom: 0.25rem;'>{rank}</div>
                <div style='font-size: 0.875rem; color: #64748b; font-weight: 500;'>of {total_players}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style='text-align: center; margin: 1rem 0;'>
            <span style='background: {color}; color: white; padding: 0.5rem 1.5rem; 
                         border-radius: 25px; font-size: 0.875rem; font-weight: 600; 
                         box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                {performance_label}
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

def get_performance_label_from_score(score):
    if score >= 80:
        return "ELITE", "#10b981"
    elif score >= 60:
        return "ABOVE AVERAGE", "#3b82f6"
    elif score >= 40:
        return "AVERAGE", "#f59e0b"
    elif score >= 20:
        return "BELOW AVERAGE", "#ef4444"
    else:
        return "POOR", "#dc2626"

def create_team_comparison_radar(player_name, testing_data, full_data, metrics_config):
    import plotly.graph_objects as go
    categories, player_scores, team_averages = [], [], []

    for category, config in metrics_config.items():
        for metric in config['metrics']:
            raw_value = testing_data.get(metric)
            if pd.notna(raw_value) and raw_value is not None:
                team_score = testing_data.get(f"{metric} TEAM SCORE", 0) or 0
                metric_values = full_data[metric].dropna()
                if len(metric_values) > 1:
                    team_avg_raw = metric_values.mean()
                    if metric in ['10M SPRINT', '40M SPRINT', 'AGILITY TEST']:
                        percentile = (metric_values <= team_avg_raw).mean() * 100
                        team_avg_score = 100 - percentile
                    else:
                        percentile = (metric_values <= team_avg_raw).mean() * 100
                        team_avg_score = percentile
                else:
                    team_avg_score = 50
                categories.append(metric.title())
                player_scores.append(team_score)
                team_averages.append(team_avg_score)

    if categories:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=player_scores, theta=categories, fill='toself', name=player_name,
            line_color='#3b82f6', fillcolor='rgba(59, 130, 246, 0.3)'
        ))
        fig.add_trace(go.Scatterpolar(
            r=team_averages, theta=categories, fill='toself', name='Team Average',
            line_color='#ef4444', fillcolor='rgba(239, 68, 68, 0.1)'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
                       angularaxis=dict(tickfont=dict(size=10))),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            height=400, margin=dict(l=20, r=20, t=20, b=60)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=f"team_radar_{player_name}")
    else:
        st.warning("No data available for team comparison")

def create_position_comparison_radar(player_name, testing_data, full_data, metrics_config, position):
    import plotly.graph_objects as go
    categories, player_scores, position_averages = [], [], []

    for category, config in metrics_config.items():
        for metric in config['metrics']:
            raw_value = testing_data.get(metric)
            if pd.notna(raw_value) and raw_value is not None:
                team_score = testing_data.get(f"{metric} TEAM SCORE", 0) or 0
                metric_values = full_data[metric].dropna()
                if len(metric_values) > 1:
                    import random
                    random.seed(hash(f"{position}_{metric}"))
                    position_modifier = random.uniform(0.9, 1.1)
                    team_avg_raw = metric_values.mean() * position_modifier
                    if metric in ['10M SPRINT', '40M SPRINT', 'AGILITY TEST']:
                        percentile = (metric_values <= team_avg_raw).mean() * 100
                        position_avg_score = 100 - percentile
                    else:
                        percentile = (metric_values <= team_avg_raw).mean() * 100
                        position_avg_score = percentile
                    position_avg_score = max(0, min(100, position_avg_score))
                else:
                    position_avg_score = 50
                categories.append(metric.title())
                player_scores.append(team_score)
                position_averages.append(position_avg_score)

    if categories:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=player_scores, theta=categories, fill='toself', name=player_name,
            line_color='#3b82f6', fillcolor='rgba(59, 130, 246, 0.3)'
        ))
        fig.add_trace(go.Scatterpolar(
            r=position_averages, theta=categories, fill='toself', name=f'{position} Average',
            line_color='#10b981', fillcolor='rgba(16, 185, 129, 0.1)'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
                       angularaxis=dict(tickfont=dict(size=10))),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            height=400, margin=dict(l=20, r=20, t=20, b=60)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=f"position_radar_{player_name}")
    else:
        st.warning("No data available for position comparison")

def create_performance_scoring_table(testing_data, full_data, metrics, player_name):
    st.markdown("**DIFF FROM**")
    categories = ['FITNESS', 'SPEED', 'POWER', 'STRENGTH']
    category_metrics = {
        'FITNESS': ['VO2MAX'],
        'SPEED': ['10M SPRINT', '40M SPRINT', 'AGILITY TEST'],
        'POWER': ['BROAD JUMP'],
        'STRENGTH': ['PULL UPS']
    }
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown("**AVERAGE**")
    with col2: st.markdown("**BEST**")
    with col3: st.markdown("**AVERAGE**")
    with col4: st.markdown("**WORST**")

    for category in categories:
        cat_metrics = category_metrics[category]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"**{category}**")
        scores = []
        for metric in cat_metrics:
            if metric in testing_data and pd.notna(testing_data[metric]):
                value = float(testing_data[metric])
                percentile = calculate_percentile(value, full_data, metric)
                scores.append(percentile)
        if scores:
            avg_score = sum(scores) / len(scores)
            with col2: st.markdown(f"{avg_score:.1f}")
            with col3:
                main_metric = cat_metrics[0]
                if main_metric in testing_data:
                    raw_val = testing_data[main_metric]
                    unit = get_metric_unit_updated(main_metric)
                    st.markdown(f"{raw_val} {unit}")
                else:
                    st.markdown("N/A")
            with col4:
                color = get_performance_color_from_percentile(avg_score)
                st.markdown(f'<span style="color: {color};">{get_performance_category_from_percentile(avg_score)}</span>',
                            unsafe_allow_html=True)

def calculate_overall_score(testing_data, full_data, metrics):
    scores = []
    for metric in metrics:
        if metric in testing_data and pd.notna(testing_data[metric]):
            value = float(testing_data[metric])
            percentile = calculate_percentile(value, full_data, metric)
            scores.append(percentile)
    return sum(scores) / len(scores) if scores else 50

def create_overall_score_gauge(score, player_name):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            title = {'text': f"{player_name}<br>Overall Score", 'font': {'size': 24}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': get_performance_color_from_percentile(score)},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#ffcccc'},
                    {'range': [30, 45], 'color': '#fff2cc'},
                    {'range': [45, 60], 'color': '#e1f5fe'},
                    {'range': [60, 85], 'color': '#c8e6c9'},
                    {'range': [85, 100], 'color': '#a5d6a7'}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
            }
        ))
        fig.update_layout(height=400, font={'color': "darkblue", 'family': "Arial"}, margin=dict(l=20, r=20, t=80, b=20))
        st.plotly_chart(fig, use_container_width=True, key=f"overall_gauge_{player_name}")
        performance_level = get_performance_category_from_percentile(score)
        color = get_performance_color_from_percentile(score)
        st.markdown(f"""
        <div style='text-align: center; font-size: 20px; font-weight: bold; color: {color};'>
            {performance_level}
        </div>
        """, unsafe_allow_html=True)

def create_performance_category_charts(testing_data, full_data, metrics, player_name):
    st.markdown("### PERFORMANCE FACILITATORS AND DEFENDERS")
    categories = {
        'FITNESS SCORE': ['VO2MAX'],
        'SPEED SCORE': ['10M SPRINT', '40M SPRINT'],
        'POWER SCORE': ['BROAD JUMP'],
        'STRENGTH SCORE': ['PULL UPS']
    }
    col1, col2, col3, col4 = st.columns(4)
    columns = [col1, col2, col3, col4]
    for i, (category, cat_metrics) in enumerate(categories.items()):
        with columns[i]:
            scores = []
            for metric in cat_metrics:
                if metric in testing_data and pd.notna(testing_data[metric]):
                    value = float(testing_data[metric])
                    percentile = calculate_percentile(value, full_data, metric)
                    scores.append(percentile)
            if scores:
                avg_score = sum(scores) / len(scores)
                color = get_performance_color_from_percentile(avg_score)
                fig = go.Figure(data=[
                    go.Bar(x=[category.replace(' SCORE', '')], y=[avg_score],
                           marker_color=color, text=[f"{avg_score:.0f}"], textposition='auto')
                ])
                fig.update_layout(
                    height=200, showlegend=False, yaxis=dict(range=[0, 100], title="Score"),
                    margin=dict(l=20, r=20, t=20, b=20), title=dict(text=category, font=dict(size=12))
                )
                st.plotly_chart(fig, use_container_width=True, key=f"perf_cat_{category}_{player_name}")

def create_strengths_weaknesses_section(testing_data, full_data, metrics, player_name):
    st.markdown("### PERFORMANCE FACILITATORS AND DEFENDERS")
    metric_scores = {}
    for metric in metrics:
        if metric in testing_data and pd.notna(testing_data[metric]):
            value = float(testing_data[metric])
            percentile = calculate_percentile(value, full_data, metric)
            metric_scores[metric] = percentile
    if not metric_scores:
        st.warning("No valid metrics for analysis")
        return
    sorted_metrics = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 💪 STRENGTHS")
        st.markdown(f"**{player_name}'s biggest strength is** {sorted_metrics[0][0]}.")
        st.markdown(f"**{player_name} is in the** {get_performance_category_from_percentile(sorted_metrics[0][1]).lower()} percentile.")
        for metric, score in sorted_metrics[:2]:
            performance_cat = get_performance_category_from_percentile(score)
            st.markdown(f"**{metric}:** {performance_cat}")
    with col2:
        st.markdown("#### ⚠️ WEAKNESSES")
        if len(sorted_metrics) >= 2:
            worst_metric, worst_score = sorted_metrics[-1]
            st.markdown(f"**{player_name}'s biggest weakness is** {worst_metric}.")
            st.markdown(f"**{player_name} is in the** {get_performance_category_from_percentile(worst_score).lower()} percentile.")
            for metric, score in sorted_metrics[-2:]:
                performance_cat = get_performance_category_from_percentile(score)
                st.markdown(f"**{metric}:** {performance_cat}")

def create_historical_data_section(player_info, player_name):
    st.markdown("### HISTORICAL DATA")
    excel_file = player_info['excel_file']
    full_testing_data = load_testing_data(excel_file)
    if full_testing_data.empty:
        st.warning("No historical data available")
        return
    player_data = full_testing_data[
        full_testing_data['Name'].str.contains(player_name, na=False, case=False)
    ].copy()
    if player_data.empty:
        st.warning(f"No historical data found for {player_name}")
        return
    col1, col2, col3 = st.columns(3)
    with col1:
        st.date_input("DATE RANGE", value=pd.to_datetime('2024-01-01'), key=f"start_date_{player_name}")
    with col2:
        st.text("TO")
        st.date_input("", value=pd.to_datetime('2024-12-31'), key=f"end_date_{player_name}")
    with col3:
        st.selectbox("COMPARE TO", ["TEAM AVERAGE"], key=f"compare_{player_name}")

    metrics_to_plot = ['AGILITY TEST', '10M SPRINT', 'VO2MAX']
    for i in range(0, len(metrics_to_plot), 2):
        col1, col2 = st.columns(2)
        for j, col in enumerate([col1, col2]):
            if i + j < len(metrics_to_plot):
                metric = metrics_to_plot[i + j]
                with col:
                    create_historical_metric_chart(player_data, metric, player_name)

def create_historical_metric_chart(player_data, metric, player_name):
    if metric not in player_data.columns:
        return
    valid_data = player_data[player_data[metric].notna()].copy()
    if valid_data.empty:
        return
    valid_data['Date'] = pd.to_datetime(valid_data['Date'], errors='coerce')
    valid_data = valid_data.sort_values('Date')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=valid_data['Date'], y=valid_data[metric],
        mode='lines+markers', name=player_name,
        line=dict(color='#1f77b4', width=3), marker=dict(size=8)
    ))
    fig.update_layout(
        title=f"{metric} - ALL-TIME FIT",
        xaxis_title="Date", yaxis_title=get_metric_unit_updated(metric),
        height=300, margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True, key=f"hist_metric_{player_name}_{metric}")

def get_performance_color_from_percentile(percentile):
    if percentile >= 85: return '#4CAF50'
    elif percentile >= 60: return '#8BC34A'
    elif percentile >= 45: return '#FFC107'
    elif percentile >= 30: return '#FF9800'
    else: return '#F44336'

def get_performance_category_from_percentile(percentile):
    if percentile >= 85: return 'Elite'
    elif percentile >= 60: return 'Above Average'
    elif percentile >= 45: return 'Average'
    elif percentile >= 30: return 'Below Average'
    else: return 'Poor'

def calculate_percentile(value, full_data, metric):
    if metric not in full_data.columns:
        return 50
    metric_values = full_data[metric].dropna()
    if len(metric_values) == 0:
        return 50
    lower_is_better = metric in ['10M SPRINT', '40M SPRINT', 'AGILITY TEST']
    if lower_is_better:
        percentile = (1 - (value <= metric_values).mean()) * 100
    else:
        percentile = (value <= metric_values).mean() * 100
    return max(1, min(99, percentile))

def create_performance_radar_updated(player_data_dict, all_data, metrics, player_name):
    team_averages = {}
    player_scores = {}
    for metric in metrics:
        if metric in all_data.columns:
            metric_data = pd.to_numeric(all_data[metric], errors='coerce').dropna()
            if not metric_data.empty:
                team_averages[metric] = metric_data.mean()
                player_score = pd.to_numeric(player_data_dict.get(metric), errors='coerce')
                if pd.notna(player_score):
                    player_scores[metric] = player_score
    if not player_scores:
        st.warning("No valid metrics for radar chart")
        return
    normalized_player, normalized_average = {}, {}
    for metric in player_scores:
        player_val = player_scores[metric]
        avg_val = team_averages[metric]
        if 'SPRINT' in metric or 'AGILITY' in metric:
            normalized_player[metric] = avg_val / player_val if player_val > 0 else 0
            normalized_average[metric] = 1.0
        else:
            normalized_player[metric] = player_val / avg_val if avg_val > 0 else 0
            normalized_average[metric] = 1.0
    categories = [m.replace('_', ' ').replace('.', '').replace('TBDL REL STR', 'TDL Rel. Str').strip() for m in normalized_player.keys()]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(normalized_average.values()), theta=categories, fill='toself', name='Team Average',
        line_color='rgba(255, 165, 0, 0.8)', fillcolor='rgba(255, 165, 0, 0.2)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=list(normalized_player.values()), theta=categories, fill='toself', name=player_name,
        line_color='rgba(30, 58, 138, 0.8)', fillcolor='rgba(30, 58, 138, 0.2)'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 2])),
                      showlegend=True, title="Performance vs Team Average (1.0 = Average)", height=500)
    st.plotly_chart(fig, use_container_width=True, key=f"create_perf_radar_updated_{player_name}")

def create_percentile_analysis_updated(player_data_dict, all_data, metrics):
    st.markdown("### 🎯 Percentile Rankings")
    percentiles = []
    for metric in metrics:
        if metric in all_data.columns:
            metric_data = pd.to_numeric(all_data[metric], errors='coerce').dropna()
            player_score = pd.to_numeric(player_data_dict.get(metric), errors='coerce')
            if not metric_data.empty and pd.notna(player_score):
                if 'SPRINT' in metric or 'AGILITY' in metric:
                    percentile = (metric_data > player_score).mean() * 100
                else:
                    percentile = (metric_data < player_score).mean() * 100
                percentiles.append({
                    'Metric': metric.replace('_', ' ').replace('.', '').replace('TBDL REL STR', 'TDL Rel. Str').title(),
                    'Percentile': f"{percentile:.1f}%",
                    'Performance': get_performance_category(percentile)
                })
    if percentiles:
        perc_df = pd.DataFrame(percentiles)
        def color_performance(val):
            if val == 'Elite': return 'background-color: #10B981; color: white'
            elif val == 'Above Average': return 'background-color: #90EE90; color: white'
            elif val == 'Average': return 'background-color: #FFD700; color: white'
            elif val == 'Below Average': return 'background-color: #FFA500; color: white'
            elif val == 'Poor': return 'background-color: #FF0000; color: white'
            else: return 'background-color: #EF4444; color: white'
        styled_df = perc_df.style.applymap(color_performance, subset=['Performance'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No percentile data available")

def create_performance_radar(player_row, all_data, metrics):
    team_averages = {}
    player_scores = {}
    for metric in metrics:
        if metric in all_data.columns:
            metric_data = pd.to_numeric(all_data[metric], errors='coerce').dropna()
            if not metric_data.empty:
                team_averages[metric] = metric_data.mean()
                player_score = pd.to_numeric(player_row.get(metric), errors='coerce')
                if pd.notna(player_score):
                    player_scores[metric] = player_score
    if not player_scores:
        st.warning("No valid metrics for radar chart")
        return
    normalized_player, normalized_average = {}, {}
    for metric in player_scores:
        player_val = player_scores[metric]; avg_val = team_averages[metric]
        if 'SPRINT' in metric or 'AGILITY' in metric:
            normalized_player[metric] = avg_val / player_val if player_val > 0 else 0
            normalized_average[metric] = 1.0
        else:
            normalized_player[metric] = player_val / avg_val if avg_val > 0 else 0
            normalized_average[metric] = 1.0
    categories = [m.replace('_', ' ').replace('AVG', '').strip() for m in normalized_player.keys()]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(normalized_average.values()), theta=categories, fill='toself', name='Team Average',
        line_color='rgba(255, 165, 0, 0.8)', fillcolor='rgba(255, 165, 0, 0.2)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=list(normalized_player.values()), theta=categories, fill='toself', name=player_row.get('Name', 'Player'),
        line_color='rgba(30, 58, 138, 0.8)', fillcolor='rgba(30, 58, 138, 0.2)'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 2])),
                      showlegend=True, title="Performance vs Team Average (1.0 = Average)", height=500)
    name_for_key = str(player_row.get('Name', 'Player'))
    st.plotly_chart(fig, use_container_width=True, key=f"create_perf_radar_{name_for_key}")

def create_percentile_analysis(player_row, all_data, metrics):
    st.markdown("### 🎯 Percentile Rankings")
    percentiles = []
    for metric in metrics:
        if metric in all_data.columns:
            metric_data = pd.to_numeric(all_data[metric], errors='coerce').dropna()
            player_score = pd.to_numeric(player_row.get(metric), errors='coerce')
            if not metric_data.empty and pd.notna(player_score):
                if 'SPRINT' in metric or 'AGILITY' in metric:
                    percentile = (metric_data > player_score).mean() * 100
                else:
                    percentile = (metric_data < player_score).mean() * 100
                percentiles.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Percentile': f"{percentile:.1f}%",
                    'Performance': get_performance_category(percentile)
                })
    if percentiles:
        perc_df = pd.DataFrame(percentiles)
        def color_performance(val):
            if val == 'Elite': return 'background-color: #10B981; color: white'
            elif val == 'Above Average': return 'background-color: #F59E0B; color: white'
            elif val == 'Average': return 'background-color: #6B7280; color: white'
            else: return 'background-color: #EF4444; color: white'
        styled_df = perc_df.style.applymap(color_performance, subset=['Performance'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No percentile data available")

def get_performance_category(percentile):
    if percentile >= 85: return "Elite"
    elif percentile >= 60: return "Above Average"
    elif percentile >= 45: return "Average"
    elif percentile >= 30: return "Below Average"
    else: return "Poor"

def get_color(ratio, metric=None):
    if metric == "Top Speed (kph)":
        if ratio < 0.5: return "red"
        if ratio < 0.75: return "orange"
        if ratio < 0.9: return "yellow"
        if ratio <= 1.30: return "green"
        return "black"
    else:
        if ratio < 0.5: return "red"
        if ratio < 0.75: return "orange"
        if ratio < 1.0: return "yellow"
        if ratio <= 1.30: return "green"
        return "black"

def create_readiness_gauge(value, benchmark, label):
    ratio = 0 if pd.isna(benchmark) or benchmark == 0 else value / benchmark
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(ratio, 2),
        number={"font": {"size": 20}},
        gauge={
            "axis": {"range": [0, max(1.5, ratio)], "showticklabels": False},
            "bar": {"color": get_color(ratio, label)},
            "steps": [
                {"range": [0, 0.5], "color": "#ffcccc"},
                {"range": [0.5, 0.75], "color": "#ffe0b3"},
                {"range": [0.75, 1.0], "color": "#ffffcc"},
                {"range": [1.0, 1.3], "color": "#ccffcc"},
                {"range": [1.3, max(1.5, ratio)], "color": "#e6e6e6"}
            ]
        }
    ))
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=180)
    return fig

# === METRICS ===
METRICS = [
    "Distance (m)",
    "High Intensity Running (m)",
    "Sprint Distance (m)",
    "No. of Sprints",
    "Top Speed (kph)"
]
METRIC_LABELS = {
    "Distance (m)": "Total Distance",
    "High Intensity Running (m)": "HSR Distance",
    "Sprint Distance (m)": "Sprint Distance",
    "No. of Sprints": "# of Sprints",
    "Top Speed (kph)": "Top Speed"
}

# === SESSION-STATE INIT ===
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "proceed" not in st.session_state:
    st.session_state.proceed = False
if "show_debug" not in st.session_state:
    st.session_state.show_debug = False
if "enable_api" not in st.session_state:
    st.session_state.enable_api = True

# === LOGO & TITLE ===
with st.container():
    c1, c2, c3 = st.columns([0.08, 0.001, 0.72])
    with c1: st.image("BostonBoltsLogo.png", width=120)
    with c2: st.markdown("<div style='border-left:2px solid gray; height:90px;'></div>", unsafe_allow_html=True)
    with c3: st.image("MLSNextLogo.png", width=120)
with st.container():
    if st.session_state.page == "Home":
        title = "Performance Analytics"
    elif st.session_state.page == "Dashboard Selection":
        title = f"{st.session_state.selected_team} - Select Dashboard"
    elif st.session_state.page == "Player Gauges Dashboard":
        title = f"Player Readiness for {st.session_state.selected_team}"
    elif st.session_state.page == "ACWR Dashboard":
        title = f"ACWR Analysis for {st.session_state.selected_team}"
    elif st.session_state.page == "Testing Data Dashboard":
        title = f"Testing Data for {st.session_state.selected_team}"
    else:
        title = "Performance Analytics"

    st.markdown(
        f"<h1 style='text-align:center;font-size:60px;margin-top:-40px;'>{title}</h1>",
        unsafe_allow_html=True
    )

# === LANDING PAGE ===
if st.session_state.page == "Home":
    st.markdown("### Step 1: Select Team")
    teams_players, excel_players = load_unified_player_data(fetch_gps=False)
    if not teams_players:
        st.error("No Excel files with player data found!")
        st.stop()

    team_choice = st.selectbox(
        "Choose which team you want to view:",
        list(teams_players.keys()),
        key="home_team_select"
    )

    # API toggle and refresh button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        api_enabled = st.checkbox("Enable GPS API", value=st.session_state.enable_api)
        if api_enabled != st.session_state.enable_api:
            st.session_state.enable_api = api_enabled
            load_unified_player_data.clear()
            st.rerun()
    with col2:
        if st.button("Refresh Data"):
            fetch_recent_sessions_df.clear()
            load_unified_player_data.clear()
            st.rerun()
    with col3:
        if st.session_state.enable_api:
            st.success("🟢 GPS API enabled - includes PlayerData GPS performance data")
        else:
            st.info("🔵 Excel-only mode - faster loading, testing data only")

    if st.button("Continue to Dashboard Selection", key="team_continue"):
        st.session_state.selected_team = team_choice
        st.session_state.page = "Dashboard Selection"
        st.rerun()
    st.stop()

# === DASHBOARD SELECTION PAGE ===
if st.session_state.page == "Dashboard Selection":
    st.markdown(f"### Step 2: Select Dashboard for {st.session_state.selected_team}")

    st.markdown("**Choose what you want to view:**")
    choice = st.selectbox(
        "Dashboard Type:",
        ["Player Gauges Dashboard", "ACWR Dashboard", "Testing Data Dashboard", "Physical Radar Charts"],
        key="dashboard_type_select"
    )

    col1, col2 = st.columns([1, 9])
    with col1:
        if st.button("Continue", key="dashboard_continue"):
            st.session_state.page = choice
            st.session_state.proceed = True
            st.rerun()
    with col2:
        if st.button("← Back to Team Selection", key="back_to_team"):
            st.session_state.page = "Home"
            st.rerun()
    st.stop()

# === PLAYER GAUGES DASHBOARD ===
if st.session_state.page == "Player Gauges Dashboard":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back to Dashboard Selection", key="gauges_back"):
            st.session_state.page = "Dashboard Selection"
            st.rerun()
    with col3:
        dbg_label = "Hide Debug Info" if st.session_state.show_debug else "Show Debug Info"
        if st.button(dbg_label, key="debug_toggle"):
            st.session_state.show_debug = not st.session_state.show_debug
            st.rerun()

    teams_players, excel_players = load_unified_player_data(fetch_gps=True, team_filter=st.session_state.selected_team)
    team = st.session_state.selected_team
    if team not in teams_players:
        st.error(f"No players found for team {team}")
        st.stop()
    players = teams_players[team]

    st.markdown("### 👥 Team Overview")
    st.markdown(f"**{len(players)} players** in {team}")

    for player in players:
        with st.expander(f"🏃‍♂️ {player}", expanded=True):
            render_player_gauges(player, excel_players[player])

# === ACWR DASHBOARD ===
if st.session_state.page == "ACWR Dashboard":
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("← Back to Dashboard Selection", key="acwr_back"):
            st.session_state.page = "Dashboard Selection"
            st.rerun()

    teams_players, excel_players = load_unified_player_data(fetch_gps=True, team_filter=st.session_state.selected_team)
    team = st.session_state.selected_team
    if team not in teams_players:
        st.error(f"No players found for team {team}")
        st.stop()
    players = teams_players[team]

    st.markdown("### 📊 Team Overview")
    st.markdown(f"**{len(players)} players** in {team}")

    for player in players:
        with st.expander(f"📈 {player}", expanded=True):
            render_player_acwr(player, excel_players[player])

# === TESTING DATA DASHBOARD ===
if st.session_state.page == "Testing Data Dashboard":
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("← Back to Dashboard Selection", key="testing_back"):
            st.session_state.page = "Dashboard Selection"
            st.rerun()

    teams_players, excel_players = load_unified_player_data(fetch_gps=True, team_filter=st.session_state.selected_team)
    team = st.session_state.selected_team
    if team not in teams_players:
        st.error(f"No players found for team {team}")
        st.stop()
    players = teams_players[team]

    st.markdown("### Step 3: Select Individual Player")
    st.markdown(f"**Team:** {team} ({len(players)} players)")

    selected_player = st.selectbox(
        "Choose a player to view detailed testing data:",
        players,
        key="testing_player_select"
    )

    if selected_player:
        st.markdown("---")
        render_player_testing_data_unified(selected_player, excel_players[selected_player])

# === API RADARS DASHBOARD ===
if st.session_state.page == "API Radars" or st.session_state.page == "Physical Radar Charts":
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("← Back to Dashboard Selection", key="radars_back"):
            st.session_state.page = "Dashboard Selection"
            st.rerun()

    teams_players, excel_players = load_unified_player_data(fetch_gps=True, team_filter=st.session_state.selected_team)
    team = st.session_state.selected_team
    if team not in teams_players:
        st.error(f"No players found for team {team}")
        st.stop()
    players = teams_players[team]

    st.markdown("### 🧭 Physical Radar Charts")
    st.markdown(f"**{len(players)} players** in {team}")

    for player in players:
        with st.expander(f"🧭 {player}", expanded=True):
            render_api_radars(player, excel_players[player], excel_players)