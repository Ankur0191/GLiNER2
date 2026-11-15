"""Rivalmap-oriented GLiNER2.5 extraction helpers.

This module is an application-layer recipe. It does not alter GLiNER2 model
weights or the upstream extraction API. It provides a deliberately compact,
non-overlapping entity taxonomy and typed relationship schema for lifestyle,
travel, premium banking, hospitality and concierge news.

Install: pip install "gliner2[local]"
Run:    python examples/rivalmap_lifestyle_extraction.py
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from gliner2 import AutoExtractor
from gliner2.joint_ie import JointIE, JointIEConfig


MODEL_ID = "fastino/gliner2.5-base-v1"

# Keep the taxonomy mutually exclusive by defining entities by their role in
# the story. Do not use both company/organization or service/business_service.
ENTITY_DESCRIPTIONS: Dict[str, str] = {
    "organization": (
        "A specifically named legal or operating organisation: company, bank, "
        "financial institution, card issuer, airline, hotel group, travel agency, "
        "concierge provider, payment network, insurer, wealth manager or corporate. "
        "Do not extract generic groups such as 'banks', 'customers' or 'family offices'."
    ),
    "brand": (
        "A specifically named commercial brand or sub-brand, distinct from the "
        "legal organisation operating it. Include hotel, airline, retail, luxury "
        "and financial brands when used as a brand name."
    ),
    "person": (
        "A named natural person, such as an executive, founder, director, spokesperson "
        "or public figure. Never classify a company or service name as a person."
    ),
    "offering": (
        "A specifically named customer-facing product, credit/debit card, paid "
        "membership, concierge programme, commercial package or named service "
        "offering. Do not extract generic service activities here."
    ),
    "loyalty_program": (
        "A specifically named loyalty, rewards, frequent-flyer or points programme. "
        "Exclude payment-card tiers, generic benefits, membership packages unless "
        "they are explicitly a points/rewards programme, and generic requirements."
    ),
    "service": (
        "A concrete service activity delivered to a customer: travel management, "
        "concierge assistance, restaurant reservations, airport meet-and-assist, "
        "lounge access, chauffeur transfers, private aviation coordination, "
        "hotel booking, yacht charter, event access or employee support. "
        "Do not extract customer groups or company names."
    ),
    "customer_segment": (
        "A customer or audience group described in the text: affluent or high-net-worth "
        "individuals, premium-card holders, private-banking clients, family offices "
        "as buyers, corporate travellers, employees or luxury leisure guests. "
        "Do not extract a named supplier or service."
    ),
    "location": (
        "A named country, city, region, airport, hotel destination or other geographic "
        "place. Do not extract an organisation merely because its name contains a place."
    ),
    "date": (
        "An explicit calendar date, month, year, or clearly stated time period."
    ),
    "monetary_value": (
        "An explicit financial amount, investment, contract value, revenue, budget, "
        "transaction value or funding amount, preserving currency and scale."
    ),
}

# Relation endpoint types are constrained. This helps prevent relations such as
# customer_segment -> provides_service or person -> operates_programme.
RELATION_DEFINITIONS: Tuple[Tuple[str, str, str, str], ...] = (
    ("partners_with", "organization", "organization",
     "An explicitly announced strategic, commercial or operating partnership."),
    ("acquired", "organization", "organization",
     "An explicit acquisition or purchase of one organisation by another."),
    ("invested_in", "organization", "organization",
     "An explicit investment by one organisation into another."),
    ("operates", "organization", "offering",
     "An organisation operates or administers a named customer offering."),
    ("provides_service", "organization", "service",
     "An organisation delivers or provides a service to customers."),
    ("offers", "organization", "offering",
     "An organisation offers, issues or makes a named product or programme available."),
    ("targets", "organization", "customer_segment",
     "An organisation explicitly targets or serves a customer segment."),
    ("serves", "offering", "customer_segment",
     "A named product or offering is designed for or serves a customer segment."),
    ("located_in", "organization", "location",
     "An organisation has an explicitly stated headquarters, office or operating location."),
    ("appointed", "organization", "person",
     "An organisation appoints or names a person to a role."),
)

# Small amount of deterministic cleanup. This is intentionally conservative:
# it canonicalises whitespace and exact duplicates, not entity identity.
GENERIC_ENTITY_VALUES = {
    "organization": {
        "company", "companies", "bank", "banks", "hotel group", "hotel groups",
        "concierge provider", "concierge providers", "travel agencies",
        "family office", "family offices", "corporate clients",
    },
    "brand": set(),
    "person": set(),
    "offering": set(),
    "loyalty_program": set(),
    "service": set(),
    "customer_segment": set(),
    "location": set(),
    "date": set(),
    "monetary_value": set(),
}


@dataclass
class ExtractionOutput:
    text: str
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    feasible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "entities": self.entities,
            "relations": self.relations,
            "feasible": self.feasible,
        }


def _normalise_surface(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t\r\n,;:. ")


def _overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return max(a["start"], b["start"]) < min(a["end"], b["end"])


def _entity_key(entity: Dict[str, Any]) -> Tuple[str, int, int]:
    return (entity["type"], entity["start"], entity["end"])


def normalise_entities(raw: Any, text: str) -> List[Dict[str, Any]]:
    """Convert JointResult/dict output into validated, de-duplicated entities."""
    if hasattr(raw, "entities") and not isinstance(raw, dict):
        values = [
            {
                "id": e.id, "type": e.type, "text": e.text,
                "start": e.start, "end": e.end,
                "confidence": e.confidence,
            }
            for e in raw.entities
        ]
    elif isinstance(raw, dict):
        values = raw.get("entities", [])
        if isinstance(values, dict):
            # Support extract_entities result shape as a convenience.
            values = [
                {"type": label, **item} if isinstance(item, dict)
                else {"type": label, "text": item}
                for label, items in values.items()
                for item in (items if isinstance(items, list) else [items])
            ]
    else:
        values = []

    cleaned: List[Dict[str, Any]] = []
    seen = set()
    for item in values:
        label = str(item.get("type", "")).strip()
        surface = _normalise_surface(str(item.get("text", "")))
        if label not in ENTITY_DESCRIPTIONS or not surface:
            continue
        if surface.casefold() in GENERIC_ENTITY_VALUES.get(label, set()):
            continue
        start, end = item.get("start"), item.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            # A text-only entity is retained but has no fabricated offsets.
            record = {"type": label, "text": surface}
        else:
            if start < 0 or end > len(text) or end <= start:
                continue
            # The source span is authoritative. Reject malformed offsets rather
            # than silently "repairing" a potentially wrong alignment.
            if text[start:end] != item.get("text"):
                continue
            record = {"type": label, "text": item.get("text"), "start": start, "end": end}
        confidence = item.get("confidence")
        if isinstance(confidence, (int, float)):
            record["confidence"] = float(confidence)
        key = (label, record.get("start"), record.get("end"), record["text"].casefold())
        if key not in seen:
            seen.add(key)
            cleaned.append(record)

    # Resolve duplicate exact spans assigned multiple labels by a stable policy.
    # Never suppress nested spans (e.g. airport within a longer venue name).
    priority = {name: i for i, name in enumerate(ENTITY_DESCRIPTIONS)}
    by_span: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    without_span = []
    for entity in cleaned:
        if "start" in entity:
            by_span.setdefault((entity["start"], entity["end"]), []).append(entity)
        else:
            without_span.append(entity)

    resolved = []
    for span_entities in by_span.values():
        # Prefer offering over loyalty_program only when the same exact span is
        # ambiguous; programmes with their own dedicated label retain that label.
        span_entities.sort(key=lambda e: (
            -float(e.get("confidence", 0.0)), priority.get(e["type"], 999)
        ))
        resolved.append(span_entities[0])

    return sorted(resolved + without_span, key=lambda e: (
        e.get("start", 10**12), e.get("end", 10**12), e["type"]
    ))


def build_joint_schema(joint: JointIE) -> Any:
    schema = joint.create_schema()
    schema.entities(ENTITY_DESCRIPTIONS)
    for name, head, tail, description in RELATION_DEFINITIONS:
        schema.relation(
            name,
            head=head,
            tail=tail,
            description=description,
            allow_self=False,
        )
    schema.no_self_loops()
    return schema


def extract_article(
    joint: JointIE,
    schema: Any,
    text: str,
    *,
    config: Optional[JointIEConfig] = None,
) -> ExtractionOutput:
    raw = joint.extract(text, schema, config=config or JointIEConfig(
        optimizer="beam",
        beam_size=32,
        candidate_threshold=0.05,
        relation_role_threshold=0.05,
        top_k_entities=48,
        top_k_roles=16,
        relation_pair_cap=192,
        max_edges_per_type=128,
        include_confidence=True,
        include_spans=True,
    ))
    if not raw.feasible:
        return ExtractionOutput(text=text, entities=[], relations=[], feasible=False)

    entities = normalise_entities(raw, text)
    id_map: Dict[str, Dict[str, Any]] = {
        e.id: {"type": e.type, "text": e.text, "start": e.start, "end": e.end}
        for e in raw.entities
    }
    relations = []
    seen_edges = set()
    for rel in raw.relations:
        head = id_map.get(rel.head)
        tail = id_map.get(rel.tail)
        if not head or not tail or head["type"] == "person" and tail["type"] == "person":
            continue
        edge_key = (rel.type, rel.head, rel.tail)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        relations.append({
            "type": rel.type,
            "head": head,
            "tail": tail,
            "confidence": rel.confidence,
        })

    return ExtractionOutput(text=text, entities=entities, relations=relations)


def extract_entities_fallback(model: Any, text: str) -> ExtractionOutput:
    """Entity-only fallback for checkpoints without a trained relation head.

    The fallback intentionally emits no relationships. It never fabricates
    edges from co-occurrence or from a guessed subject/object pairing.
    """
    raw = model.extract_entities(
        text,
        ENTITY_DESCRIPTIONS,
        include_confidence=True,
        include_spans=True,
    )
    entities = normalise_entities(raw, text)
    return ExtractionOutput(text=text, entities=entities, relations=[])


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Rivalmap entities and relations from lifestyle news.")
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model ID or local checkpoint path")
    parser.add_argument("--text", help="Article text. When omitted, read stdin.")
    parser.add_argument("--entities-only", action="store_true", help="Use ordinary entity extraction; no relation graph.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    args = parser.parse_args()

    if args.text:
        text = args.text
    else:
        import sys
        text = sys.stdin.read().strip()
    if not text:
        parser.error("Provide --text or pipe article text on stdin.")

    if args.entities_only:
        model = AutoExtractor.from_pretrained(args.model)
        output = extract_entities_fallback(model, text)
    else:
        joint = JointIE.from_pretrained(args.model)
        schema = build_joint_schema(joint)
        output = extract_article(joint, schema, text)

    print(json.dumps(output.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
