# Experiments

Each experiment gets its own dedicated subfolder here (e.g.
`experiments/ag-news/`) with a runnable script plus a short README. Run
output goes to a `runs/` subfolder inside the experiment's own directory
(gitignored — see `.gitignore`); the script and README themselves are
tracked.

| Experiment | What it exercises |
|---|---|
| [`ag-news/`](ag-news/README.md) | The `experiment.py` `induce`/`classify` loop against a real HuggingFace dataset ([`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news)) — general-domain news topic classification |
| [`trec/`](trec/README.md) | Same loop against [`SetFit/TREC-QC`](https://huggingface.co/datasets/SetFit/TREC-QC) — intent/question-type classification |
| [`pubmed-rct/`](pubmed-rct/README.md) | Same loop against [`armanc/pubmed-rct20k`](https://huggingface.co/datasets/armanc/pubmed-rct20k) — domain-specific, scientific-register sentence-role classification |
