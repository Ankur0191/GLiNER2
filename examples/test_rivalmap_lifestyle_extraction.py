"""Fast, dependency-light tests for the Rivalmap lifestyle extraction helpers."""
from examples.rivalmap_lifestyle_extraction import (
    ENTITY_DESCRIPTIONS,
    GENERIC_ENTITY_VALUES,
    normalise_entities,
)


def test_taxonomy_has_no_company_organization_or_business_service_overlap():
    assert "company" not in ENTITY_DESCRIPTIONS
    assert "business_service" not in ENTITY_DESCRIPTIONS
    assert "organization" in ENTITY_DESCRIPTIONS
    assert "service" in ENTITY_DESCRIPTIONS


def test_normaliser_rejects_generic_organisation_mentions():
    text = "The bank expanded its concierge desk."
    raw = {
        "entities": {
            "organization": [
                {"text": "bank", "start": 4, "end": 8, "confidence": 0.99}
            ]
        }
    }
    assert normalise_entities(raw, text) == []


def test_normaliser_preserves_valid_offsets_and_confidence():
    text = "Collinson expanded Priority Pass access."
    raw = {
        "entities": {
            "organization": [
                {"text": "Collinson", "start": 0, "end": 9, "confidence": 0.98}
            ],
            "offering": [
                {"text": "Priority Pass", "start": 18, "end": 31, "confidence": 0.91}
            ],
        }
    }
    result = normalise_entities(raw, text)
    assert [e["text"] for e in result] == ["Collinson", "Priority Pass"]
    assert all(text[e["start"]:e["end"]] == e["text"] for e in result)


def test_normaliser_rejects_broken_offsets_instead_of_rewriting_them():
    text = "Ten Lifestyle Group partners with GTC."
    raw = {
        "entities": {
            "organization": [
                {"text": "Ten Lifestyle Group", "start": 0, "end": 19, "confidence": 0.9}
            ]
        }
    }
    assert normalise_entities(raw, text) == []


def test_generic_categories_are_not_returned_as_supplier_entities():
    assert "family offices" in GENERIC_ENTITY_VALUES["organization"]
    assert "international business travellers" not in GENERIC_ENTITY_VALUES["organization"]
