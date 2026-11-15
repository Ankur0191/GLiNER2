import json
from pathlib import Path

import pytest

from rivalmap_ie.profiles import available_profiles, load_profile, merge_profile


def test_expected_profiles_are_available():
    assert {"base", "lifestyle", "testing_inspection_certification"} <= set(available_profiles())


@pytest.mark.parametrize("name", ["lifestyle", "testing_inspection_certification"])
def test_profile_merges_base_entities_and_domain_relations(name):
    profile = merge_profile(name)
    assert "organization" in profile["entity_descriptions"]
    assert "person" in profile["entity_descriptions"]
    assert profile["business_lines"]
    assert profile["event_types"]
    assert profile["relations"]


def test_lifestyle_has_concierge_and_travel_concepts():
    profile = merge_profile("lifestyle")
    service = profile["entity_descriptions"]["service"].lower()
    assert "concierge" in service
    assert "lounge access" in service
    assert "restaurant" in service


def test_tic_has_testing_inspection_and_standards():
    profile = merge_profile("testing_inspection_certification")
    assert "standard" in profile["entity_descriptions"]
    assert "certification" in profile["entity_descriptions"]
    assert "laboratory" in profile["entity_descriptions"]["service"].lower()
    assert any(relation["name"] == "inspects" for relation in profile["relations"])


def test_unknown_profile_fails_clearly():
    with pytest.raises(FileNotFoundError):
        load_profile("nonexistent_future_industry")


def test_profile_json_is_valid():
    for name in available_profiles():
        json.loads((Path(__file__).parents[1] / "profiles" / f"{name}.json").read_text())
