# Rivalmap Lifestyle Information Extraction

This folder contains an application-layer recipe for using GLiNER2.5 on news
about premium banking, cardholder benefits, luxury travel, concierge,
hospitality, loyalty, experiences and corporate lifestyle services.

It does **not** fine-tune model weights. It improves the extraction setup by
using descriptive, non-overlapping labels, typed relationship endpoints,
conservative output validation and explicit handling of checkpoints that do
not expose a relation head.

## Install

From the repository root:

```powershell
pip install -e ".[local,test]"
```

## Run

Entity-only (works with an entity-capable GLiNER2.5 checkpoint):

```powershell
python -m examples.rivalmap_lifestyle_extraction --entities-only --text "Collinson expanded Priority Pass lounge access for premium cardholders."
```

Entities + typed graph (requires GLiNER2.5 checkpoint trained with relations):

```powershell
python -m examples.rivalmap_lifestyle_extraction --text "Global Travel Collection partnered with Ten Lifestyle Group to provide concierge services to premium travellers."
```

Pipe a complete article:

```powershell
Get-Content .\article.txt -Raw | python -m examples.rivalmap_lifestyle_extraction
```

## Design choices

- `organization` absorbs legal companies and operating organisations. Do not
  split the same supplier between `company` and `organization`.
- `brand` is separate from the legal operator when the article distinguishes them.
- `offering` covers named products/programmes/packages; `loyalty_program`
  is reserved for named points/rewards/frequent-flyer programmes.
- `service` covers activities; `customer_segment` covers audiences.
- Relationship endpoints are typed in `JointSchema`; this is preferable to
  independent relation tuples when building a counterparty graph.
- The entity-only fallback returns no inferred relations. Co-occurrence does
  not prove partnership, ownership, operation or provision.
- Confidence is a model score, not a calibrated probability. Tune thresholds
  against a manually annotated real-news benchmark before production use.

## Current limitations

The quality of each label still depends on checkpoint training. A schema cannot
teach a model facts it has never learned, and application-layer validation cannot
turn a model prediction into a verified fact. In particular, verify named legal
entities, brand/operator links, and event time against source documents before
writing them to Rivalmap's canonical graph.
