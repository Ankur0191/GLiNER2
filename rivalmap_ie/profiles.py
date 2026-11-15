"""Load and validate industry profiles without importing model runtimes."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
REQUIRED_RELATION_KEYS = {"name", "head", "tail", "description"}


def available_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))


def load_profile(profile: str) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9_]+", profile):
        raise ValueError("Profile names may contain lowercase letters, digits and underscores only.")
    path = PROFILE_DIR / f"{profile}.json"
    if not path.is_file():
        choices = ", ".join(available_profiles())
        raise FileNotFoundError(f"Unknown profile {profile!r}. Available profiles: {choices}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("profile_id") != profile:
        raise ValueError(f"Profile id mismatch in {path}")
    for relation in data.get("relations", []):
        missing = REQUIRED_RELATION_KEYS - set(relation)
        if missing:
            raise ValueError(f"Invalid relation in {profile}: missing {sorted(missing)}")
    return data


def merge_profile(profile: str) -> dict[str, Any]:
    base = load_profile("base")
    if profile == "base":
        return base
    domain = load_profile(profile)
    result = dict(base)
    result["entity_descriptions"] = dict(base.get("entities", {}))
    result["entity_descriptions"].update(domain.get("entity_extensions", {}))
    result["relations"] = base.get("relations", []) + domain.get("relations", [])
    result["business_lines"] = domain.get("business_lines", [])
    result["event_types"] = domain.get("event_types", [])
    result["profile_id"] = domain["profile_id"]
    result["profile_version"] = domain["profile_version"]
    result["description"] = domain["description"]
    # A profile relation with the same name intentionally overrides base definition.
    dedup = {}
    for relation in result["relations"]:
        dedup[relation["name"]] = relation
    result["relations"] = list(dedup.values())
    return result
