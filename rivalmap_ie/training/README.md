# Training data

Put human-reviewed examples in this directory or another private data location.
Do not commit confidential customer documents or scraped content whose license
does not permit redistribution.

JSONL uses GLiNER2 native format. Example:

```jsonl
{"input":"Bureau Veritas inspected the offshore platform in Aberdeen.","output":{"entities":{"organization":["Bureau Veritas"],"service":["inspected"],"asset":["offshore platform"],"location":["Aberdeen"]}}}
```

This is a format illustration, not a sufficient training dataset. Build at least
a few hundred diverse reviewed examples per target profile before evaluating
fine-tuning; more may be required for robust production performance.
