#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from api_helpers import API_BASE, USER_AGENT, auth_headers, request_json


def register_agent(name: str, description: str) -> Dict[str, Any]:
    status, data = request_json(
        "POST",
        f"{API_BASE}/agents/register",
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        json_body={"name": name, "description": description},
    )
    if status >= 400:
        raise RuntimeError(f"Register failed ({status}): {data}")
    return data


def get_claim_status(api_key: str) -> str:
    status, data = request_json("GET", f"{API_BASE}/agents/status", headers=auth_headers(api_key))
    print(f"[debug] status check: HTTP {status}, data={data}")
    if status >= 400:
        raise RuntimeError(f"Status failed ({status}): {data}")
    return str(data.get("status", "unknown"))


def _deep_get(d: Any, path: List[str]) -> Optional[Any]:
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def parse_register_response(data: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
    candidates = [
        (["agent", "api_key"], ["agent", "claim_url"], ["agent", "verification_code"]),
        (["data", "agent", "api_key"], ["data", "agent", "claim_url"], ["data", "agent", "verification_code"]),
        (["api_key"], ["claim_url"], ["verification_code"]),
        (["data", "api_key"], ["data", "claim_url"], ["data", "verification_code"]),
    ]
    for kpath, cpath, vpath in candidates:
        api_key = _deep_get(data, kpath)
        claim_url = _deep_get(data, cpath)
        ver_code = _deep_get(data, vpath)
        if api_key and claim_url:
            return str(api_key), str(claim_url), (str(ver_code) if ver_code else None)
    raise RuntimeError(f"Unexpected register response shape: {data}")
