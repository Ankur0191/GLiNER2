"""Lifestyle-industry entity extraction test for GLiNER2.

IMPORTANT:
    The article fixtures below are synthetic test scenarios. They are written
    in the style of corporate/lifestyle-industry news, but they are NOT claims
    that these announcements or partnerships actually happened.

Run:
    python test_gliner_lifestyle.py

Optional Hugging Face authentication (useful for fewer download/rate-limit
issues):
    set HF_TOKEN=hf_...          # PowerShell
    python test_gliner_lifestyle.py
"""

import json
from collections import defaultdict

from gliner2 import AutoExtractor

MODEL_ID = "fastino/gliner2.5-base-v1"

# These labels cover the main entity types found in banking, premium cards,
# concierge, travel, hospitality, loyalty, and corporate-benefits news.
# `service` is included beside `business_service` to compare which wording the
# model recognises more reliably for service descriptions.
LABELS = [
    "company",
    "organization",
    "brand",
    "person",
    "product",
    "location",
    "business_service",
    "service",
    "loyalty_program",
    "financial_product",
    "customer_segment",
    "industry",
    "date",
    "monetary_value",
]

# Synthetic fixtures featuring a mix of global and regional industry entities.
# Real company/brand names are used as recognition anchors; the described news
# events, launches, contracts, metrics, and partnerships are fictional.
ARTICLES = [
    {
        "id": "LIFE-001",
        "section": "Banking & premium cardholder concierge",
        "headline": "Northstar Bank expands its premium cardholder concierge programme",
        "body": """
Northstar Bank has announced a proposed expansion of its premium lifestyle
benefits for holders of its World Elite Mastercard and Northstar Reserve cards.
Under the fictional programme scenario, the bank is working with Ten Lifestyle
Group to provide 24-hour cardholder concierge support across India, Singapore,
the United Arab Emirates and the United Kingdom.

The service package includes restaurant reservations, luxury hotel bookings,
private aviation coordination, event access and airport meet-and-assist.
Priya Mehta, Head of affluent banking at Northstar Bank, said the initiative is
designed for high-net-worth individuals, private banking clients and frequent
international travellers. The bank's internal planning note allocates
US$4.8 million for the first 18 months, with a proposed launch in Mumbai,
Dubai, Singapore and London in March 2027.
""",
    },
    {
        "id": "LIFE-002",
        "section": "Travel management & travel benefits",
        "headline": "Global Travel Collection targets growth in managed luxury travel",
        "body": """
Global Travel Collection, part of Internova Travel Group, is preparing a new
premium travel-management proposition for independent travel advisors and
corporate clients. The fictional announcement describes a service integration
with Amex GBT covering complex itinerary planning, executive travel support,
VIP airport assistance and disruption management.

The proposition is aimed at chief executives, family offices, entertainment
professionals and multinational companies with frequent-travel requirements.
Jennifer Wilson, President of Global Travel Collection, said clients increasingly
expect a single point of contact for air travel, private transfers, villa stays
and last-minute itinerary changes. The test brief references New York, London,
Los Angeles, Hong Kong and Dubai as priority markets, with an estimated
€12 million technology and operations investment over two years.
""",
    },
    {
        "id": "LIFE-003",
        "section": "Airport experiences & card benefits",
        "headline": "Aster Financial reviews airport experience benefits for affluent customers",
        "body": """
Aster Financial is reviewing the airport benefits attached to its Aster
Infinite credit card, a fictional premium financial product issued on the
Visa network. The proposed benefits package combines Priority Pass lounge
access, airport fast-track services, chauffeur transfers and a 24/7 travel
assistance desk operated by Collinson.

The review focuses on affluent mass-market customers, wealth-management
clients and international business travellers. Aster Financial's Chief
Customer Officer, Rahul Kapoor, said the company wants to connect payments,
travel protection and in-person airport experiences in one card proposition.
The internal scenario compares service coverage at Indira Gandhi International
Airport in New Delhi, Chhatrapati Shivaji Maharaj International Airport in
Mumbai, Singapore Changi Airport and Heathrow Airport in London. The proposed
annual benefits budget is INR 320 crore for the 2027 financial year.
""",
    },
    {
        "id": "LIFE-004",
        "section": "Luxury hospitality & experiences",
        "headline": "Four Seasons and a fictional wealth platform package bespoke stays",
        "body": """
Meridian Private Wealth is developing a bespoke hospitality programme that
would give eligible clients access to selected Four Seasons properties and
curated destination experiences. The fictional concept includes luxury hotel
reservations, private dining, yacht charters, wellness retreats and
personalised family itineraries.

The service is intended for ultra-high-net-worth families, wealth advisers and
family-office executives. Four Seasons Hotel Mumbai, Four Seasons Resort
Maldives at Kuda Huraa and Four Seasons Hotel London at Park Lane are named as
illustrative destinations in the planning material. Elena Rossi, Meridian's
Global Head of Client Experience, said the proposal is designed to extend the
relationship beyond investment advice into lifestyle and travel needs.
The pilot is pencilled in for the second quarter of 2027.
""",
    },
    {
        "id": "LIFE-005",
        "section": "Airlines, loyalty & rewards",
        "headline": "Aurora Airways considers a new premium tier for frequent flyers",
        "body": """
Aurora Airways is considering a new invitation-only tier within its Aurora
Privilege loyalty programme. The fictional proposal would add guaranteed
premium-cabin upgrade requests, dedicated disruption support, chauffeur
transfers and access to selected hospitality experiences.

The airline is assessing possible collaborations with Marriott Bonvoy,
Accor Live Limitless and Mastercard Priceless. The target audience includes
business-class passengers, high-frequency corporate travellers and members
with substantial annual spend. Daniel Reed, Chief Commercial Officer at
Aurora Airways, said the programme should recognise both travel frequency and
long-term customer value. The research team is using London, Doha, Singapore,
Delhi and Sydney as sample hubs and has modelled an incremental rewards budget
of US$22 million per year.
""",
    },
    {
        "id": "LIFE-006",
        "section": "Luxury concierge & lifestyle management",
        "headline": "Concierge providers compete for family-office and luxury-client mandates",
        "body": """
Quintessentially and Sienna Charles are cited in a fictional industry briefing
on the growing demand for outsourced lifestyle management. The briefing
identifies private members' club access, restaurant reservations, art advisory,
private jet charter coordination, yacht bookings, household staffing and
bespoke event planning as common client requirements.

The report says family offices, founders, senior executives and ultra-high-net-
worth individuals increasingly expect concierge providers to coordinate
multiple suppliers through one relationship manager. It also highlights
Blacklane for premium ground transportation and VistaJet for private aviation
as examples of specialist brands operating in the wider luxury-services
ecosystem. The scenario estimates that a large family office could spend
between US$150,000 and US$500,000 annually on external lifestyle services.
""",
    },
    {
        "id": "LIFE-007",
        "section": "Employee concierge & corporate benefits",
        "headline": "Cedar Group pilots employee lifestyle support across three markets",
        "body": """
Cedar Group, a fictional multinational employer, is planning an employee
concierge and corporate-benefits pilot with Pluxee. The proposed employee
wellbeing package includes relocation assistance, family travel planning,
restaurant and event reservations, childcare-resource guidance and local
experience recommendations.

The pilot is designed for senior managers, internationally mobile employees
and staff relocating with families. Maria Fernandes, Chief People Officer at
Cedar Group, said the programme is intended to reduce the administrative
burden associated with relocation and improve the employee experience.
The test scenario covers Bengaluru, Gurugram and Singapore, with a proposed
budget of INR 7.5 crore for 4,000 eligible employees during calendar year 2027.
""",
    },
    {
        "id": "LIFE-008",
        "section": "Hospitality partnerships & loyalty ecosystems",
        "headline": "Hotel group explores experience-led rewards with payment partners",
        "body": """
The fictional Seabrook Hotels Group is evaluating a rewards proposition that
would connect hotel stays with dining, wellness and local cultural experiences.
The concept references IHG One Rewards, Hilton Honors and World of Hyatt as
examples of established hotel loyalty programmes, and Visa Infinite and
American Express Membership Rewards as examples of premium payments and
rewards ecosystems.

Potential benefits include late checkout, room upgrades, spa appointments,
chef-led dining, airport transfers and access to destination events. Seabrook's
Chief Growth Officer, Oliver Grant, said the group is assessing whether a
single experience wallet could improve repeat bookings among luxury leisure
guests, premium-card holders and small-business owners. The planning document
lists Bali, Tokyo, Paris, New York and Cape Town as candidate destinations.
""",
    },
]


def print_article_result(article_id, section, headline, result):
    print("\n" + "=" * 100)
    print(f"{article_id} | {section}")
    print(headline)
    print("=" * 100)

    entities = result.get("entities", {}) if isinstance(result, dict) else {}
    found_any = False

    for label in LABELS:
        values = entities.get(label, [])
        if not values:
            continue
        found_any = True
        print(f"\n{label}:")
        for item in values:
            if isinstance(item, dict):
                entity_text = item.get("text", str(item))
                confidence = item.get("confidence")
                if confidence is None:
                    print(f"  - {entity_text}")
                else:
                    print(f"  - {entity_text}  [confidence: {confidence:.3f}]")
            else:
                print(f"  - {item}")

    if not found_any:
        print("No entities returned. Check the model output below.")
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def main():
    print(f"Loading GLiNER2 model: {MODEL_ID}")
    model = AutoExtractor.from_pretrained(MODEL_ID)
    print("Model loaded successfully.")
    print(f"Testing {len(ARTICLES)} synthetic lifestyle-industry news articles...")

    totals = defaultdict(int)
    unique_entities = defaultdict(set)

    for article in ARTICLES:
        article_text = (
            f"{article['headline']}\n\n"
            f"Industry section: {article['section']}\n\n"
            f"{article['body'].strip()}"
        )

        result = model.extract_entities(
            article_text,
            LABELS,
            include_confidence=True,
        )

        print_article_result(
            article["id"],
            article["section"],
            article["headline"],
            result,
        )

        entities = result.get("entities", {}) if isinstance(result, dict) else {}
        for label, values in entities.items():
            if not isinstance(values, list):
                continue
            totals[label] += len(values)
            for item in values:
                value = item.get("text") if isinstance(item, dict) else str(item)
                if value:
                    unique_entities[label].add(value.strip())

    print("\n\n" + "#" * 100)
    print("CORPUS SUMMARY")
    print("#" * 100)
    for label in LABELS:
        if totals[label] or unique_entities[label]:
            print(
                f"{label:22} mentions={totals[label]:3}  "
                f"unique={len(unique_entities[label]):3}"
            )
            for entity in sorted(unique_entities[label], key=str.casefold):
                print(f"  - {entity}")

    print("\nTest completed.")
    print("Reminder: all article events and relationships in this test corpus are fictional.")


if __name__ == "__main__":
    main()
