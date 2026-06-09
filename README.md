# evorare

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/hinanohart/evorare/actions/workflows/ci.yml/badge.svg)](https://github.com/hinanohart/evorare/actions/workflows/ci.yml)

**evorare** is a CPU / offline diagnostic instrument that describes the
*realized-sample diversity trend* of an LLM evolutionary-search archive
(FunSearch / AlphaEvolve / ShinkaEvolve / OpenEvolve) using ecology
**Hill numbers** + **Rao quadratic entropy**, and routes which estimators are
statistically valid for *this* archive before they are allowed to drive an
early-stopping decision.

> **What it is — and is not.**
> evorare **describes realized-sample diversity only; does not estimate population diversity.**
> The default featurizers measure **syntactic/structural diversity, not semantic diversity**.
> It is a **diagnostic instrument, not a stopping guarantee.**
> All headline numbers come from **synthetic** ground-truth scenarios with known labels.

No model calls. No GPU. Core dependency: **numpy** only.

## Install

```bash
pip install evorare            # core (numpy only)
pip install "evorare[bench]"   # + optional framework adapters
```

## Usage

```bash
evorare diagnose archive.jsonl --featurizer behavior,ast --out result.json --svg out.svg
evorare gate                   # run sensitivity gates G1..G9 (exit 0/1)
evorare convert openevolve <checkpoint_dir> -o archive.jsonl   # optional adapter
```

Input is generic JSON-Lines (one record per line):

```json
{"id": "p17", "code": "def f(): ...", "score": 0.83, "generation": 4, "parent_id": "p9", "island_id": 1}
```

Required fields: `id`, `code`, `score`. Optional: `generation`, `parent_id`, `island_id`.

## What it measures

- **Primary stopping signal:** the generation trend of the **Hill q1 effective number** and
  **Rao quadratic entropy** (both descriptive statistics of the realized sample).
- **Sampling-validity routing:** under strong selection the Good-Turing coverage and Chao
  estimators are *not* valid; evorare detects this and excludes them from the stopping decision.
- **Genealogy (only when `parent_id` is present):** lineage survivorship and genealogical
  mean-pairwise-distance, computed exactly from the known parent links.

### Validation on synthetic ground truth

All numbers below are produced by `evorare gate` / `python scripts/run_bench.py` and stored in
[`results/v0.1.0a1_metrics.json`](results/v0.1.0a1_metrics.json) (seed=0, 20 seeds/scenario,
8 generations, 120-resample bootstrap). They are **synthetic** ground-truth checks, not real
framework results.

| Sensitivity gate (G1–G9, all passing) | Result |
|---|---|
| G2 — separate growing vs depleting diversity (Hill q1 trend, AUC) | **1.00** |
| G3 — locate a planted lineage bottleneck within ±2 generations | **1.00** |
| G3 — false lineage-collapse on a stationary (turnover-only) archive | **0.00** |
| G8 — exclude coverage when selection breaks it (S-AGGREGATED) | **1.00** |
| G8 — false exclusion under uniform sampling (S-NULL) | **0.00** |

The test suite has **89 tests** (`pytest`), all passing on Ubuntu and Windows for Python
3.10–3.12.

> The Hill q1 *slope* point estimate is a plug-in value; its confidence interval is a bootstrap
> percentile interval and may not be centred on the point estimate, because the Hill q1 plug-in
> is downward-biased under finite-sample resampling (a known property of diversity estimators).
> The verdict uses the interval, which is the conservative, statistically meaningful quantity.

## Prior art & honest limits

See [docs/limits.md](docs/limits.md). evorare builds on established ecology estimators
(**Chao**, **rarefaction**, Hill numbers, Rao Q) and population-genetics lineage statistics;
the contribution is their transplant to LLM evolutionary-search archives plus the
sampling-validity routing. Prior diversity management inside **FunSearch** / AlphaEvolve uses
island models and behavior descriptors; evorare is a framework-agnostic external diagnostic.

## License

MIT — see [LICENSE](LICENSE).
