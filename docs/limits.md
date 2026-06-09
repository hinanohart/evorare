# Honest limits, scope, and prior art

evorare is a **measurement instrument**, not a model and not a stopping guarantee. This page
states what it does not do, so the headline is not over-read.

## What evorare does not claim

1. **It describes realized-sample diversity only; it does not estimate population diversity.**
   Selection bias distorts the *population* interpretation of a diversity value, not the value
   itself. evorare reports the realized-sample descriptors (Hill numbers, Rao Q) as trends, and
   routes the population-extrapolating estimators (Chao point estimates, iNEXT) out of the
   decision because the archive violates their random-sampling assumption.

2. **The default featurizers measure syntactic/structural diversity, not semantic diversity.**
   The behaviour featurizer bins by score (phenotype); the AST featurizer captures program
   *shape*. An optional embedding featurizer (`[embed]`, local model, still offline) adds a
   semantic axis but is **experimental** and not validated here.

3. **Coverage and Chao are auxiliary, never a sole stopping signal.** Good-Turing coverage
   assumes random independent sampling; under selection (a few high-fitness species resampled
   repeatedly — statistically the same as spatial aggregation) it over-states completeness. The
   sampling-validity gate detects this with an index-of-dispersion test and excludes those
   estimators. The primary signals are the Hill q1 and Rao Q trends.

4. **All headline numbers are synthetic.** They come from labelled generators whose correct
   verdict is known by construction (`evorare gate`). Real framework archives are supported as
   input but are exploratory: this release does not claim validation on real archives.

5. **Genealogy is exact, not inferred — and optional.** Lineage survivorship and genealogical
   mean-pairwise-distance are computed directly from known `parent_id` links. There is no
   probabilistic ancestry model and no effective-population-size estimate. When `parent_id` is
   absent the genealogy axis is skipped (the verdict degrades to two values), never fabricated.

6. **It diagnoses; it does not improve search.** evorare reports a trend-based verdict
   (HEALTHY / SATURATING / GENEALOGY-COLLAPSE / INDETERMINATE / DESCRIPTIVE). It does not tune
   the search and does not guarantee that stopping now is optimal.

## Prior art

The estimators are established ecology and population-genetics tools:

- **Hill numbers / rarefaction / extrapolation** — Chao et al. 2014; Hsieh et al. (iNEXT) 2016.
- **Chao1/Chao2 richness estimators and their log-normal CIs** — Chao 1987; Chao & Jost 2012.
- **Rao quadratic entropy** — Rao 1982 (pairwise-distance-weighted diversity).

Diversity management already exists *inside* evolutionary-search systems — FunSearch's
island model, AlphaEvolve's MAP-Elites grid — but it is framework-internal and typically uses
edit/Hamming distance or behaviour descriptors, not the ecology estimator family. evorare's
contribution is the **transplant of those estimators to LLM evolutionary-search archives as a
framework-agnostic external diagnostic, together with the sampling-validity routing** that
decides which estimators are valid for a given archive. The narrow band is exactly that
combination; broad "measure diversity" framing is not claimed.

## Determinism & platforms

Species ids use blake2b (not Python's randomised `hash()`), all bootstraps are seeded, ancestor
walks are iterative, and file I/O is UTF-8 — so results are byte-reproducible across processes
and on Windows as well as Linux.
