"""Token bootstrapping for a freshly started Label Studio instance.

Pure helpers (urls/tokens/credentials in, working token out) — no ORM, no
config. Ported verbatim from the old fleet.py.

On first boot we try the token we generated and passed via
``LABEL_STUDIO_USER_TOKEN`` (Path A); if the image ignores it we log in and
read the user's legacy token (Path B).
"""

import time

import requests


def token_works(ls_url: str, token: str) -> bool:
    try:
        response = requests.get(
            f"{ls_url}/api/projects?page_size=1",
            headers={"Authorization": f"Token {token}"},
            timeout=5,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def wait_until_http(ls_url: str, *, timeout: int = 180):
    """Wait until the instance answers HTTP. First boot runs DB migrations and
    can take well over a minute, so this is patient."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            requests.get(ls_url, timeout=5)
            return
        except requests.RequestException:
            time.sleep(3)
    raise RuntimeError(f"Label Studio did not start at {ls_url}")


def login_session(ls_url: str, email: str, password: str) -> requests.Session:
    session = requests.Session()
    login_url = f"{ls_url}/user/login/"
    session.get(login_url, timeout=10)
    csrf = session.cookies.get("csrftoken", "")
    session.post(
        login_url,
        data={"email": email, "password": password, "csrfmiddlewaretoken": csrf},
        headers={"Referer": login_url},
        timeout=10,
    )
    return session


def enable_legacy_tokens(ls_url: str, email: str, password: str):
    """LS >= 1.23 disables `Token <token>` auth by default; this codebase uses
    it, so turn it back on. No-op on versions without the endpoint."""
    try:
        session = login_session(ls_url, email, password)
        csrf = session.cookies.get("csrftoken", "")
        session.post(
            f"{ls_url}/api/jwt/settings",
            json={"api_tokens_enabled": True, "legacy_api_tokens_enabled": True},
            headers={"X-CSRFToken": csrf, "Referer": ls_url},
            timeout=10,
        )
    except requests.RequestException:
        pass


def fetch_token_via_login(ls_url: str, email: str, password: str) -> str:
    """Path B: establish a session, then read the user's legacy token."""
    session = login_session(ls_url, email, password)
    response = session.get(f"{ls_url}/api/current-user/token", timeout=10)
    response.raise_for_status()
    return response.json()["token"]


def resolve_token(ls_url: str, *, candidate_token: str, email: str, password: str) -> str:
    """Return a working token, trying the pre-set one first (Path A -> Path B)."""
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if token_works(ls_url, candidate_token):
            return candidate_token
        try:
            fetched = fetch_token_via_login(ls_url, email, password)
            if fetched and token_works(ls_url, fetched):
                return fetched
        except requests.RequestException:
            pass
        time.sleep(3)

    raise RuntimeError(
        f"Could not obtain a working API token from {ls_url}. "
        "The container may still be migrating, or legacy tokens may be disabled."
    )
