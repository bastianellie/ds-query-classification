# Experiments

Each experiment gets its own dedicated subfolder here (e.g.
`experiments/ag-news/`) with a runnable script plus a short README. Run
output goes to a `runs/` subfolder inside the experiment's own directory
(gitignored — see `.gitignore`); the script and README themselves are
tracked.

| Experiment | What it exercises |
|---|---|
| [`ag-news/`](ag-news/README.md) | The `experiment.py` `induce`/`classify` loop against a real HuggingFace dataset ([`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news)) |
