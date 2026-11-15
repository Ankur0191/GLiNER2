from rivalmap_ie.extract import extract


class FakeExtractor:
    def extract_entities(self, text, schema, **kwargs):
        return {
            "entities": {
                "organization": [
                    {"text": "Bureau Veritas", "start": 0, "end": 14, "confidence": 0.98},
                    {"text": "Bureau Veritas", "start": 0, "end": 14, "confidence": 0.97},
                ],
                "service": [
                    {"text": "inspect", "start": 15, "end": 22, "confidence": 0.91},
                    {"text": "not in article", "start": 99, "end": 113, "confidence": 0.9},
                ],
            }
        }


def test_entity_only_mode_deduplicates_and_rejects_invalid_offsets(monkeypatch):
    import gliner2

    monkeypatch.setattr(gliner2.AutoExtractor, "from_pretrained", lambda *_args, **_kwargs: FakeExtractor())
    text = "Bureau Veritas inspect"
    result = extract("fake-model", "testing_inspection_certification", text, entities_only=True)
    assert len(result["entities"]) == 2
    assert result["relations"] == []
    assert result["relation_extraction"] == "disabled_entity_only_mode"
    assert result["entities"][0]["text"] == "Bureau Veritas"
