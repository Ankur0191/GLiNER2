# Rivalmap Generic Industry Intelligence Extraction

A profile-driven application layer over GLiNER2.5 for structured extraction
from news, company announcements, regulatory updates and industry documents.
The core is industry-neutral; JSON profiles add domain vocabulary, business
lines, event types and typed relationships.

## Built-in profiles

- `base`: universal corporate intelligence entities/relationships.
- `lifestyle`: premium banking, concierge, luxury travel, loyalty, hospitality and experiences.
- `testing_inspection_certification`: testing, inspection, certification, assurance, audits, verification and conformity assessment.

## Install

```powershell
pip install -e ".[local,train,test]"
```

## Run extraction

```powershell
python -m rivalmap_ie.extract --profile lifestyle --text "Global Travel Collection partnered with Ten Lifestyle Group to launch a white-label concierge service."
python -m rivalmap_ie.extract --profile testing_inspection_certification --text "Bureau Veritas won a contract to inspect offshore wind turbine foundations under an accredited inspection scope."
```

For article files:

```powershell
Get-Content .\article.txt -Raw | python -m rivalmap_ie.extract --profile testing_inspection_certification --output .\article.json
```

The selected profile controls the extraction schema. It does not guarantee that
a generic checkpoint already recognizes every specialist term. Review outputs
and annotate errors.

## Fine-tuning workflow

The repo's trainer supports JSONL and LoRA. Fine-tuning is a separate stage from
profile-driven inference. Do not train directly on model-generated predictions
without human review.

1. Create a reviewed gold JSONL file using the labels and descriptions in the
   selected profile. Keep the original article as `input`.
2. Each `output.entities` mention must occur verbatim in the input. Use empty
   lists for negative labels in examples where those types are absent.
3. Use the same label definitions across all records for a training run.
4. Split by article into train/validation; do not randomly split sentences from
   one article across both sets.
5. Train one shared adapter on a balanced mix of domains, or keep independent
   adapters per domain. The latter is recommended initially to prevent one
   industry's vocabulary from interfering with another.
6. Benchmark on held-out, human-annotated examples before promoting an adapter.

Prepare reviewed training data:

```powershell
python -m rivalmap_ie.training.prepare_dataset --input data\raw\lifestyle_gold.jsonl --output-dir data\processed\lifestyle
```

Train a LoRA adapter:

```powershell
python -m rivalmap_ie.training.train_lora --train data\processed\lifestyle\train.jsonl --validation data\processed\lifestyle\validation.jsonl --output-dir models\lifestyle_adapter
```

For TIC replace lifestyle data/output paths with the TIC equivalents.

## Add another industry

Copy `profiles/base.json` to `profiles/<industry>.json`, then add
`entity_extensions`, typed `relations`, `business_lines` and `event_types`.
No Python source edits are needed for schema-only profiles.

## Important constraints

- The profiles are schemas, not trained weights.
- The seed corpus includes examples for testing and annotation guidance; those
  examples are illustrative only and must not be used as factual reporting.
- Relation extraction requires a checkpoint with a trained relation head.
  If the checkpoint lacks one, use the existing entity-only recipe rather than
  asserting graph edges from co-occurrence.
- A confidence score is not a calibrated probability.
- Treat extracted evidence, model interpretation and strategic inference as
  separate fields in downstream Rivalmap records.
