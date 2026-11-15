"""Generic profile-driven extraction CLI using GLiNER2."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from rivalmap_ie.profiles import available_profiles, merge_profile


def build_joint_schema(joint: Any, profile: dict[str, Any]):
    schema = joint.create_schema()
    schema.entities(profile["entity_descriptions"])
    for rel in profile["relations"]:
        schema.relation(
            rel["name"],
            head=rel["head"],
            tail=rel["tail"],
            description=rel["description"],
            allow_self=False,
        )
    schema.no_self_loops()
    return schema


def extract(model_id: str, profile_name: str, text: str, entities_only: bool = False) -> dict[str, Any]:
    profile = merge_profile(profile_name)

    if entities_only:
        from gliner2 import AutoExtractor

        model = AutoExtractor.from_pretrained(model_id)
        raw = model.extract_entities(
            text,
            profile["entity_descriptions"],
            include_confidence=True,
            include_spans=True,
        )
        raw_entities = raw.get("entities", []) if isinstance(raw, dict) else []
        if isinstance(raw_entities, dict):
            raw_entities = [
                {"type": label, **item} if isinstance(item, dict)
                else {"type": label, "text": item}
                for label, mentions in raw_entities.items()
                for item in (mentions if isinstance(mentions, list) else [mentions])
            ]
        entities, seen = [], set()
        for entity in raw_entities:
            label = entity.get("type")
            surface = entity.get("text")
            if label not in profile["entity_descriptions"] or not isinstance(surface, str) or not surface:
                continue
            start, end = entity.get("start"), entity.get("end")
            item = {"type": label, "text": surface}
            if isinstance(start, int) and isinstance(end, int):
                if start < 0 or end <= start or text[start:end] != surface:
                    continue
                item.update(start=start, end=end)
            confidence = entity.get("confidence")
            if isinstance(confidence, (int, float)):
                item["confidence"] = float(confidence)
            key = (label, item.get("start"), item.get("end"), surface.casefold())
            if key not in seen:
                seen.add(key)
                entities.append(item)
        return {
            "profile": profile_name,
            "profile_version": profile["profile_version"],
            "feasible": True,
            "entities": entities,
            "relations": [],
            "relation_extraction": "disabled_entity_only_mode",
            "business_lines": profile.get("business_lines", []),
            "event_types": profile.get("event_types", []),
        }

    from gliner2.joint_ie import JointIE, JointIEConfig

    joint = JointIE.from_pretrained(model_id)
    schema = build_joint_schema(joint, profile)
    result = joint.extract(
        text,
        schema,
        config=JointIEConfig(
            optimizer="beam",
            beam_size=32,
            candidate_threshold=0.05,
            relation_role_threshold=0.05,
            top_k_entities=64,
            top_k_roles=24,
            relation_pair_cap=256,
            max_edges_per_type=128,
            include_confidence=True,
            include_spans=True,
        ),
    )
    if not result.feasible:
        return {
            "profile": profile_name,
            "profile_version": profile["profile_version"],
            "feasible": False,
            "entities": [],
            "relations": [],
            "business_lines": profile.get("business_lines", []),
            "event_types": profile.get("event_types", []),
        }

    id_map = {
        entity.id: {
            "type": entity.type,
            "text": entity.text,
            "start": entity.start,
            "end": entity.end,
        }
        for entity in result.entities
    }
    entities, seen_entities = [], set()
    for entity in result.entities:
        key = (entity.type, entity.start, entity.end)
        if key in seen_entities:
            continue
        seen_entities.add(key)
        item = id_map[entity.id].copy()
        confidence = getattr(entity, "confidence", None)
        if confidence is not None:
            item["confidence"] = float(confidence)
        entities.append(item)

    relations, seen_relations = [], set()
    for relation in result.relations:
        head, tail = id_map.get(relation.head), id_map.get(relation.tail)
        if not head or not tail:
            continue
        key = (relation.type, relation.head, relation.tail)
        if key in seen_relations:
            continue
        seen_relations.add(key)
        relations.append({
            "type": relation.type,
            "head": head,
            "tail": tail,
            "confidence": float(relation.confidence),
        })

    return {
        "profile": profile_name,
        "profile_version": profile["profile_version"],
        "feasible": True,
        "entities": entities,
        "relations": relations,
        "business_lines": profile.get("business_lines", []),
        "event_types": profile.get("event_types", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile-driven Rivalmap extraction.")
    parser.add_argument("--profile", default="lifestyle", choices=available_profiles())
    parser.add_argument("--model", default="fastino/gliner2.5-base-v1")
    parser.add_argument("--entities-only", action="store_true",
                        help="Use AutoExtractor and do not attempt relation extraction.")
    parser.add_argument("--text", help="Source text; stdin is used when omitted.")
    parser.add_argument("--output", help="Optional path for JSON output.")
    args = parser.parse_args()

    text = args.text if args.text is not None else sys.stdin.read()
    text = text.strip()
    if not text:
        parser.error("Provide --text or pipe article text through stdin.")

    result = extract(args.model, args.profile, text, entities_only=args.entities_only)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
