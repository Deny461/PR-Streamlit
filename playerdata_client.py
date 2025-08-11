# playerdata_client.py
import time
import random
import requests
import pandas as pd
from requests.exceptions import RequestException

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
        # Retry token fetch to handle transient 5xx/503 outages
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        last_error = None
        for attempt in range(4):
            try:
                resp = requests.post(self.token_url, data=payload, timeout=30)
                # Treat 5xx as retryable
                if resp.status_code >= 500:
                    raise RequestException(f"{resp.status_code} {resp.reason}")
                resp.raise_for_status()
                data = resp.json()
                self._token = data["access_token"]
                self._exp = time.time() + int(data.get("expires_in", 3600))
                return self._token
            except Exception as e:
                last_error = e
                # Exponential backoff with jitter
                if attempt < 3:
                    sleep_s = (0.6 * (2 ** attempt)) + random.uniform(0, 0.3)
                    time.sleep(sleep_s)
                else:
                    break
        raise RuntimeError(f"Token request failed after retries: {last_error}")

    def gql(self, query, variables=None):
        # Retry GraphQL calls on transient failures and token refresh if needed
        last_error = None
        for attempt in range(4):
            try:
                headers = {"Authorization": f"Bearer {self._bearer()}"}
                resp = requests.post(
                    self.graphql_url,
                    json={"query": query, "variables": variables or {}},
                    headers=headers,
                    timeout=60,
                )
                if resp.status_code == 401 and attempt < 3:
                    # token might be expired early; force refresh by clearing
                    self._token = None
                    continue
                if resp.status_code >= 500:
                    raise RequestException(f"{resp.status_code} {resp.reason}")
                resp.raise_for_status()
                out = resp.json()
                if "errors" in out and out["errors"]:
                    raise RuntimeError(out["errors"])
                return out["data"]
            except Exception as e:
                last_error = e
                if attempt < 3:
                    sleep_s = (0.6 * (2 ** attempt)) + random.uniform(0, 0.3)
                    time.sleep(sleep_s)
                else:
                    break
        raise RuntimeError(f"GraphQL request failed after retries: {last_error}")

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