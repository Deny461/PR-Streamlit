# playerdata_client.py
import time
import requests
import pandas as pd

class PlayerDataClient:
    """
    Minimal OAuth + GraphQL client for PlayerData.
    Uses client credentials and exposes .gql(query, variables).
    """
    def __init__(self, client_id, client_secret, token_url, graphql_url):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.graphql_url = graphql_url
        self._token = None
        self._exp = 0

    def _bearer(self):
        if self._token and time.time() < self._exp - 30:
            return self._token
        r = requests.post(self.token_url, data={
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }, timeout=30)
        r.raise_for_status()
        j = r.json()
        self._token = j["access_token"]
        self._exp = time.time() + int(j.get("expires_in", 3600))
        return self._token

    def gql(self, query, variables=None):
        headers = {"Authorization": f"Bearer {self._bearer()}"}
        r = requests.post(self.graphql_url,
                          json={"query": query, "variables": variables or {}},
                          headers=headers,
                          timeout=60)
        r.raise_for_status()
        out = r.json()
        if "errors" in out:
            raise RuntimeError(out["errors"])
        return out["data"]

def sessions_to_df(athlete_name: str, sessions: list) -> pd.DataFrame:
    """
    Shape API sessions to EXACT columns your app expects from CSVs.
    Keeps render_* logic unchanged.
    """
    rows = []
    for s in sessions:
        sess_type = "Match Session" if s.get("__typename") == "MatchSession" else "Training Session"
        ms = s.get("athleteMetricSet") or {}
        rows.append({
            "Athlete Name": athlete_name,
            "Start Date": (s.get("startTime") or "")[:10],  # 'YYYY-MM-DD'
            "Session Type": sess_type,
            "Segment Name": "Whole Session",
            "Duration (mins)": s.get("durationMin"),
            "Distance (m)": ms.get("totalDistanceM"),
            "High Intensity Running (m)": ms.get("totalHighIntensityDistanceM"),
            "Sprint Distance (m)": ms.get("totalSprintDistanceM"),
            "No. of Sprints": ms.get("sprintEvents"),
            "Top Speed (kph)": ms.get("maxSpeedKph"),
        })
    return pd.DataFrame(rows)