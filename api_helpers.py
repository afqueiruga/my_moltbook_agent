#!/usr/bin/env python3
from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from tenacity import RetryCallState, retry, retry_if_result, stop_after_attempt

# ----------------------------
# Config
# ----------------------------

API_BASE = "https://www.moltbook.com/api/v1"  # IMPORTANT: keep www (avoid redirect auth issues)
WEB_BASE = "https://www.moltbook.com"
USER_AGENT = "moltbook-theoremsprite/0.4"

# Retry behavior
MAX_RETRIES = 8
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_MAX_SECONDS = 120.0


# ----------------------------
# HTTP helpers
# ----------------------------

def normalize_submolt_name(submolt: Optional[str]) -> str:
    """
    Normalize a user-facing submolt reference to an API submolt name.

    Moltbook UI/community references are often written like "m/ai", while the API
    typically expects the bare submolt name (e.g. "ai", "general").
    """
    s = str(submolt or "").strip()
    if s.lower().startswith("m/"):
        s = s[2:]
    return s or "general"


def auth_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }


def request_json(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Tuple[int, Dict[str, Any]]:
    try:
        r = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=json_body,
            params=params,
            timeout=timeout,
            allow_redirects=False,
        )
        # Debug: check if we got redirected (301/302/307/308)
        if r.status_code in (301, 302, 307, 308):
            print(f"[debug] REDIRECT detected: {r.status_code} -> {r.headers.get('Location', 'unknown')}")
    except requests.RequestException as e:
        # 599 is a common convention for "network connect timeout error"/synthetic status.
        return 599, {"success": False, "error": f"Request failed: {e.__class__.__name__}: {e}"}
    try:
        data = r.json() if r.content else {}
    except Exception:
        data = {"success": False, "error": f"Non-JSON response: {r.text[:200]}"}
    return r.status_code, data


def sleep_with_jitter(seconds: float) -> None:
    time.sleep(seconds + random.uniform(0, 2.0))


# ----------------------------
# Tenacity retry helpers
# ----------------------------

def _is_retryable_result(result: Tuple[int, Dict[str, Any]]) -> bool:
    """Check if response status warrants a retry."""
    status, _ = result
    return status in {429, 500, 502, 503, 504, 599}


def _wait_with_retry_after(retry_state: RetryCallState) -> float:
    """
    Custom wait function that respects retry_after hints from API responses.
    Falls back to exponential backoff if no hint is present.
    """
    if retry_state.outcome and not retry_state.outcome.failed:
        status, data = retry_state.outcome.result()
        if status == 429:
            # Check for retry_after hints in response body
            for key, multiplier in [("retry_after_seconds", 1), ("retry_after_minutes", 60)]:
                val = data.get(key)
                if val is not None:
                    try:
                        wait_time = float(val) * multiplier
                        return min(wait_time, BACKOFF_MAX_SECONDS) + random.uniform(0, 2.0)
                    except (ValueError, TypeError):
                        pass

    # Exponential backoff fallback
    attempt = retry_state.attempt_number
    delay = min(BACKOFF_BASE_SECONDS * (2 ** attempt), BACKOFF_MAX_SECONDS)
    return delay + random.uniform(0, 2.0)


def _log_retry_attempt(retry_state: RetryCallState) -> None:
    """Log information about retry attempts for observability."""
    if retry_state.outcome and not retry_state.outcome.failed:
        status, data = retry_state.outcome.result()
        attempt = retry_state.attempt_number
        if status == 429:
            remaining = data.get("daily_remaining")
            msg = f"[rate-limit] attempt {attempt}/{MAX_RETRIES}"
            if remaining is not None:
                msg += f", daily_remaining={remaining}"
            print(msg)
        elif status >= 500:
            print(f"[retry] server error {status}, attempt {attempt}/{MAX_RETRIES}")


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=_wait_with_retry_after,
    retry=retry_if_result(_is_retryable_result),
    before_sleep=_log_retry_attempt,
)
def _request_with_retry(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Tuple[int, Dict[str, Any]]:
    """Low-level request with retry. Returns (status, data)."""
    return request_json(method, url, headers=headers, json_body=json_body, params=params, timeout=timeout)


def moltbook_request(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
    operation: str = "Request",
) -> Dict[str, Any]:
    """Make a Moltbook API request with retry. Raises RuntimeError on failure."""
    status, data = _request_with_retry(
        method, url, headers=headers, json_body=json_body, params=params, timeout=timeout
    )
    if status >= 400:
        raise RuntimeError(f"{operation} failed ({status}): {data}")
    return data


def get_personal_feed(api_key: str, sort: str = "new", limit: int = 10) -> Dict[str, Any]:
    print(f"[debug] fetching feed...")
    result = moltbook_request(
        "GET", f"{API_BASE}/feed",
        headers=auth_headers(api_key),
        params={"sort": sort, "limit": limit},
        operation="Feed",
    )
    print(f"[debug] feed returned {len(result.get('posts', result.get('items', [])))} items")
    return result


def create_post(api_key: str, submolt: str, title: str, content: str) -> Dict[str, Any]:
    submolt_name = normalize_submolt_name(submolt)
    return moltbook_request(
        "POST", f"{API_BASE}/posts",
        headers=auth_headers(api_key),
        json_body={"submolt": submolt_name, "title": title, "content": content},
        operation="Create post",
    )


def semantic_search(
    api_key: str,
    q: str,
    *,
    type: str = "all",
    limit: int = 20,
    submolt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Semantic search posts/comments.

    Docs: GET /search?q=...&type=posts|comments|all&limit=...
    This client also supports optional submolt filtering:
    - First tries sending `submolt=<name>` to the API.
    - If the API rejects the parameter (HTTP 400), falls back to client-side filtering.
    """
    if not q or not str(q).strip():
        raise ValueError("q must be non-empty")

    submolt_name = normalize_submolt_name(submolt) if submolt else None
    params: Dict[str, Any] = {"q": str(q), "type": str(type), "limit": int(limit)}
    if submolt_name:
        params["submolt"] = submolt_name

    status, data = _request_with_retry(
        "GET",
        f"{API_BASE}/search",
        headers=auth_headers(api_key),
        params=params,
    )

    if status < 400:
        return data

    # Fallback if server doesn't accept submolt param.
    if submolt_name and status == 400:
        params.pop("submolt", None)
        status2, data2 = _request_with_retry(
            "GET",
            f"{API_BASE}/search",
            headers=auth_headers(api_key),
            params=params,
        )
        if status2 >= 400:
            raise RuntimeError(f"Search failed ({status2}): {data2}")

        # Results may be top-level or nested under "data".
        container: Optional[Dict[str, Any]] = None
        if isinstance(data2, dict) and isinstance(data2.get("results"), list):
            container = data2
        elif isinstance(data2, dict) and isinstance(data2.get("data"), dict) and isinstance(data2["data"].get("results"), list):
            container = data2["data"]

        if container is None:
            return data2

        results = container.get("results")
        if not isinstance(results, list):
            return data2

        filtered: List[Dict[str, Any]] = []
        for r in results:
            if not isinstance(r, dict):
                continue
            sm = r.get("submolt")
            sm_name = None
            if isinstance(sm, dict):
                sm_name = sm.get("name")
            elif sm is not None:
                sm_name = sm
            if sm_name is not None and normalize_submolt_name(str(sm_name)) == submolt_name:
                filtered.append(r)

        container["results"] = filtered
        container["count"] = len(filtered)
        container["filtered_by_submolt"] = submolt_name
        return data2

    raise RuntimeError(f"Search failed ({status}): {data}")

def comment_on_post(api_key: str, post_id: str, content: str, parent_id: Optional[str] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"content": content}
    if parent_id:
        body["parent_id"] = parent_id
    url = f"{API_BASE}/posts/{post_id}/comments"
    hdrs = auth_headers(api_key)
    print(f"[debug] POST {url}")
    print(f"[debug] headers: Authorization={hdrs.get('Authorization', 'MISSING')[:25]}...")
    return moltbook_request(
        "POST", url,
        headers=hdrs,
        json_body=body,
        operation="Comment",
    )


def get_agent_posts(api_key: str, limit: int = 10) -> Dict[str, Any]:
    """Get the agent's own posts via the profile endpoint."""
    print(f"[debug] fetching own posts via /agents/me ...")
    result = moltbook_request(
        "GET", f"{API_BASE}/agents/me",
        headers=auth_headers(api_key),
        operation="Get own profile+posts",
    )
    # The profile endpoint returns recentPosts at top level (per docs),
    # but some deployments may nest under "agent" or "data".
    recent = result.get("recentPosts")
    if recent is None and isinstance(result.get("agent"), dict):
        recent = result["agent"].get("recentPosts")
    if recent is None and isinstance(result.get("data"), dict):
        recent = result["data"].get("recentPosts")
    if recent is None:
        recent = []
    if not isinstance(recent, list):
        print(f"[debug] /agents/me recentPosts unexpected type: {type(recent).__name__}")
        recent = []
    print(f"[debug] found {len(recent)} own posts")
    return {"posts": recent[:limit]}


def get_post_comments(api_key: str, post_id: str, *, sort: str = "new") -> Dict[str, Any]:
    """Get comments on a specific post."""
    print(f"[debug] fetching comments for post {post_id}...")
    result = moltbook_request(
        "GET", f"{API_BASE}/posts/{post_id}/comments",
        params={"sort": sort},
        headers=auth_headers(api_key),
        operation=f"Get comments for post {post_id}",
    )
    print(f"[debug] found {len(result.get('comments', []))} comments")
    return result
