"""Smoke-test a realistic synthetic lifestyle-industry news article.

Synthetic scenario; company/brand names may be real, but the announced event,
deal terms and quotes below are fictional.
"""
from examples.rivalmap_lifestyle_extraction import (
    MODEL_ID,
    build_joint_schema,
    extract_article,
)
from gliner2.joint_ie import JointIE, JointIEConfig

ARTICLE = """
Global Travel Collection (GTC), the luxury travel division of Internova Travel
Group, has entered a strategic partnership with Ten Lifestyle Group to expand
premium concierge support for selected clients in Asia-Pacific and the Middle
East. Under the agreement, Ten Lifestyle Group will provide 24/7 lifestyle
management, restaurant reservations, private aviation coordination, luxury
hotel bookings and access to major sporting and cultural events.

The programme is intended for high-net-worth individuals, family offices and
premium-card holders. GTC's partner travel advisors will coordinate complex
international itineraries, while Ten Lifestyle Group will manage lifestyle
requests and local experiences. The companies said the pilot will begin in
Singapore, Hong Kong and Dubai in the second quarter of 2027. The financial
terms were not disclosed.
"""


def main():
    joint = JointIE.from_pretrained(MODEL_ID)
    schema = build_joint_schema(joint)
    result = extract_article(
        joint, schema, ARTICLE,
        config=JointIEConfig(
            optimizer="beam",
            beam_size=32,
            top_k_entities=48,
            top_k_roles=16,
            candidate_threshold=0.05,
            relation_role_threshold=0.05,
            include_confidence=True,
            include_spans=True,
        ),
    )
    print(result.to_dict())


if __name__ == "__main__":
    main()
