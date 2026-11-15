# Training data

Put human-reviewed examples in this directory or another private data location.
Do not commit confidential customer documents or scraped content whose license
does not permit redistribution.

JSONL uses GLiNER2 native format. Example:

```jsonl
{"input":"Bureau Veritas inspected the offshore platform in Aberdeen.","output":{"entities":{"organization":["Bureau Veritas"],"service":["inspected"],"asset":["offshore platform"],"location":["Aberdeen"]}}}
```

This is a format illustration, not a sufficient training dataset. Start with
hundreds of reviewed and varied articles per profile, then increase coverage
based on held-out error analysis. The actual quantity needed depends on task
complexity and annotation consistency.

Use explicit empty lists for labels absent from each example. Use the same
label definitions across all examples in a training run. The provided dataset
preparer validates entity mentions/spans, groups records by source URL or
article ID when metadata exists, and creates a leakage-aware split.

A generated or model-predicted label is not a gold annotation until a reviewer
has confirmed it against the source.
