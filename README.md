# LLM Perspective Coverage on Perspectrum

Measures whether an LLM's generated opinions represent the **full spread** of human
perspectives on a claim (from the Perspectrum dataset), rather than collapsing onto
a single stance.

## Pipeline

1. **Data prep** — parse Perspectrum into `claims.csv` + `perspectives.csv`.
2. **Generation** — sample `k` outputs per claim from the LLM configured in
   `configs/config.yaml` (currently `mistralai/Mistral-7B-Instruct-v0.3`).
   Claims are decoded in batches of `generation.batch_size` (default 10)
   claims per `model.generate()` call — all `batch_size × k` sequences are
   generated together, which keeps the GPU busy on real batches instead of
   near-idle single-claim decodes (~10x wall-clock speedup; lower
   `batch_size` if a larger model OOMs).
3. **Embeddings** — embed human perspectives and LLM outputs with `BAAI/bge-large-en-v1.5`.
4. **Calibration** — empirically validate the coverage similarity threshold τ against
   labeled pairs (ROC/PR analysis) instead of assuming a fixed value.
5. **Analysis** — compute per-claim coverage, stance balance, distribution divergence,
   and intra-LLM diversity.
6. **Notebooks** — visualize and summarize results.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You'll also need:
- A Hugging Face account + access token for the model set in
  `configs/config.yaml` under `llm.model` (currently `mistralai/Mistral-7B-Instruct-v0.3`).
- Set the token: `export HF_TOKEN=your_token_here`
- A GPU is strongly recommended for local generation — `llm.provider: huggingface`
  loads the model 4-bit quantized on GPU (falls back to float32 on CPU, which will
  be slow). If you don't have a GPU, set `llm.provider: groq` in
  `configs/config.yaml`, pick a model available in your Groq account, and set
  `GROQ_API_KEY` instead — no local compute needed.

## Getting the real Perspectrum data

This repo ships with a **small synthetic sample** (`data/raw/perspectrum/*.json`)
with the exact same schema as the real dataset, so the whole pipeline runs
end-to-end out of the box for testing.

To use the real dataset:
1. Go to https://github.com/CogComp/perspectrum
2. Download `dataset_split_v1.0.json`, `perspective_pool_v1.0.json`,
   `evidence_pool_v1.0.json`
3. Replace the sample files in `data/raw/perspectrum/` with the real ones
   (same filenames).
4. Re-run `python run_pipeline.py --stage data_prep`

## Running the full pipeline

```bash
python run_pipeline.py                  # runs every stage
python run_pipeline.py --stage generation   # runs just one stage
python run_pipeline.py --model 12b          # pick a ladder model (1.5b/3b/7b/12b/32b)
python -m pytest tests/ -q                  # unit tests (parser, metrics, client plumbing)
```

Stages: `data_prep`, `generation`, `embeddings`, `calibration`, `analysis`

## Output

- `results/metrics/per_claim_coverage.csv` — coverage score per claim
- `results/metrics/generation_quality.csv` — per-claim quality gate: calls,
  usable viewpoints, parse-ok rate, and calls that yielded zero usable
  viewpoints (also rolled into `final_report_summary.json`)
- `results/metrics/stance_distribution.csv` — human vs LLM stance balance
  (support/undermine/unclear ratios)
- `results/metrics/llm_sample_stances.csv` — per-viewpoint stance: the LLM's
  self-reported stance alongside the embedding-NN predicted stance
- `results/metrics/stance_crosscheck.csv` + `stance_agreement.json` —
  confusion matrix and agreement rate of self-reported vs predicted stance
  (cross-check that the classifier driving coverage/divergence agrees with
  what the model said about itself; also rolled into `final_report_summary.json`
  as `stance_agreement_rate`)
- `results/metrics/divergence_scores.csv` — JS divergence per claim
- `results/metrics/threshold_calibration.json` — ROC-AUC/PR-AUC and the
  empirically recommended τ (see "Threshold validation" below)
- `results/metrics/threshold_calibration_pairs.csv` — every labeled
  (LLM output, human perspective) pair used to calibrate τ, with its judge
  label and cosine similarity
- `results/final_report_summary.json` — aggregate numbers across the dataset
- `results/figures/*.png` — plots, including `threshold_roc_pr.png`
  (run via notebook 04 or `analysis/make_figures.py` / `analysis/threshold_calibration.py`)

## Key metric: coverage score

For each claim, and for a similarity threshold τ:

```
coverage(claim) = |{ human perspectives p : max_i sim(LLM_sample_i, p) >= τ }|
                  ------------------------------------------------------------
                              total human perspectives for claim
```

A coverage score near 1.0 means the LLM's samples, collectively, touched on
every distinct human perspective. A low score means the LLM is only
reflecting a subset (often just one side) of the real opinion spread.

## Threshold validation (fixes the "arbitrary τ" gap)

τ is no longer just assumed. `python -m src.analysis.threshold_calibration`:

1. Samples `calibration.n_llm_samples` (default 50) LLM outputs and pairs
   each with the human perspectives on the *same* claim (this is the
   distribution the coverage metric actually compares against — including
   both same-stance and opposing-stance pairs — rather than random
   cross-claim pairs, which would trivially separate near zero similarity).
2. Labels each pair "same perspective" (1) or not (0) via a strict-criteria
   LLM-as-judge (`calibration.judge_mode: llm`, the default - a fixed
   external Groq judge with reasoning disabled, deliberately not one of the
   generation subjects so no arm is self-judged) or exports the
   pairs for a human to label (`calibration.judge_mode: manual`, then re-run
   with `--score-only`).
3. Computes BGE cosine similarity for every pair and plots the ROC and
   Precision-Recall curves of similarity vs. label.
4. Reports the Youden's-J-optimal and F1-optimal thresholds, saved to
   `results/metrics/threshold_calibration.json` alongside ROC-AUC/PR-AUC.

`coverage_score.py` (and `summarize_results.py`) automatically use the
F1-optimal calibrated τ once this has been run, and fall back to the
`analysis.similarity_threshold` config default only when no calibration
exists yet — logging a loud warning that the fallback is an unvalidated
heuristic. Set `analysis.use_calibrated_threshold: false` to force the
config default regardless.

## Structured generation (fixes the "compression information loss" gap)

The generation prompt (`llm.prompt_template` in `configs/config.yaml`) no
longer asks for a single compressed sentence. It neutrally asks what the
model thinks about the claim and requests structured JSON - exactly ONE
canonical shape (bare objects, one per stance the model holds, never an
array), anchored by a tiny two-stance example. An earlier "object OR array"
wording reliably made small models emit degenerate empty arrays, so the
array alternative was removed:

```json
{"stance": "support", "perspective_summary": "2-3 sentences ...", "core_reasoning": "1-2 sentences ..."}
{"stance": "undermine", "perspective_summary": "...", "core_reasoning": "..."}
```

`src/generation/postprocess_generations.py` parses this (tolerating markdown
fences and minor formatting drift, and falling back to raw-text-as-summary
with `parse_ok=False` if parsing fails, so degraded samples are tracked
rather than silently corrupted or dropped) and builds `combined_text =
perspective_summary + core_reasoning`. `src/embeddings/embed_llm_outputs.py`
embeds `combined_text`, not the old single-sentence `generated_text` — so
coverage now measures the LLM's actual perspective space rather than the
side effect of a one-sentence compression prompt. The self-reported
`llm_reported_stance` is also carried through to the embedding index for
cross-checking against the embedding-nearest-neighbor stance classifier.
