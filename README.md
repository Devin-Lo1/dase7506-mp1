# DASE7506 MP1 - frozen submission

Student ID: **3036802959**. Original score submission: **#19**.
Full-test CPU FP32 score: **1.4381909205827006 BPB**. Validation: 1.4217038295231355.
Final training: 1,200 updates, seed 17, effective batch 512, 157,286,400 targets.
The <1.4 development goal was not achieved. The model was frozen before testing;
there has been no test-based tuning. See **report/MP1_Report.pdf** (7 pages).

## 1. Install and evaluate without retraining

Download and extract both archives side by side:

```
MP1_code/   # this repository, report and benchmark files
MP1_model/  # frozen checkpoint plus baseline for verification
```

Use Python 3.12 (original environment: 3.12.14, macOS arm64). All commands run
from `MP1_code/`. Dependencies are pinned; no FAISS package is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_submission.py --checkpoint ../MP1_model/checkpoint.pt
python evaluate.py --checkpoint ../MP1_model/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test --output runs/reproduced_test.json
```

Expected `bpb`: **1.4381909205827006**. Floating-point differences across hardware
may cause small variations. Evaluation is offline after installation. All learned
inference data live in the checkpoint; the included `data/` is the unmodified
teacher benchmark, with original hashes and tokenizer. The original CPU scorer
is unchanged. MPS is used only for training, not ranked evaluation.

## 2. Reproduce resource measurement

Close other training/scoring processes first. The baseline checkpoint is provided
for comparison only and is not an inference dependency of the submitted model.

```bash
python measure_validation.py --candidate ../MP1_model/checkpoint.pt --baseline ../MP1_model/baseline_checkpoint.pt --split test --repeats 3 --output-dir runs/reproduced_resources
```

Original full-test measurements: 4.672703x baseline CPU median time, 2.580750 GiB
maximum RSS, 58.568878 MiB uncompressed inference files. Limits: 5x / 4 GiB / 64 MiB.
The wrapper supports macOS/Linux and reports RSS using `getrusage`. It invokes the
unchanged teacher scorer in fresh subprocesses, alternating baseline and candidate.
Reviewers should measure the ratio on their own machine. No absolute laptop timing
is claimed for other systems. `measure_validation.py` calls its pass field
`development_limits_pass` even for test; the `split` field records `test` correctly.

## 3. Reproduce training from random initialization (optional)

Retraining is NOT needed to reproduce the submitted score. The exact original
training command used Apple MPS with BF16 computation and FP32 master parameters:

```bash
python train.py --implementation student_local --config configs/student_local_wide.json --device mps --precision bf16 --threads 4 --seed 17 --steps 1200 --batch-size 64 --accumulation-steps 8 --learning-rate .004 --min-lr-ratio 0 --warmup-steps 60 --optimizer muon --weight-decay .3 --log-every 25 --eval-every 300 --run-dir runs/retrain-wide
```

A CPU alternative uses `--device cpu --precision fp32`; this may be much slower
and is not expected to reproduce the stochastic training trajectory bit for bit.
Even on MPS, a seed alone does not guarantee bitwise deterministic retraining.
New run directories are required. Final score reproduction uses the supplied bundle.

Build the final count tensors using training text only:

```bash
python build_ngram.py --min-count 2 --output runs/counts/base.pt
python refine_count_asset.py --base runs/counts/base.pt --output runs/counts/full-trigram-kn5.pt
python assemble_final.py --neural runs/retrain-wide/checkpoint-step-1200.pt --counts runs/counts/full-trigram-kn5.pt --output runs/rebuilt-final/checkpoint.pt
```

This streamlined count construction skips discarded orders 6-12; its tensor values
are checked against every count tensor in the frozen checkpoint during packaging.
Metadata and checkpoint file hashes differ when assets are rebuilt. `assemble_final.py`
uses the already-frozen global settings in `configs/final_hybrid.json`; it performs
no new fitting, search, or test-based model selection.

Original baseline recipe:

```bash
python train.py --implementation model --config configs/baseline.json --device cpu --precision fp32 --threads 4 --seed 17 --steps 1200 --batch-size 32 --learning-rate .001 --warmup-steps 100 --weight-decay .1 --run-dir runs/retrain-baseline
```

The final recipe uses 16x the baseline's gradient targets. A matched-target neural
control at step 75 (9,830,400 targets) scored 2.026129 validation BPB versus
2.071087 for the baseline; this is a whole-recipe comparison, not a matched
architecture-only or learning-rate-schedule ablation.

## 4. Tests and evidence

```bash
python -m unittest discover -s tests -v
```

`evidence/runs/` contains compact original metrics, search results, failure records,
freeze records, resource runs and ablations. Their historical local paths identify
provenance and do not need to exist on the reviewing computer. No cached per-target
validation/test probabilities or intermediate development weights are deployed.
`evidence/runs/final-1200/test_cpu_fp32.json` is the official score record.
`evidence/runs/final-ablations/summary.json` records component-removal diagnostics.
`EXPERIMENTS.md` preserves development history, including unsuccessful experiments.
Historical commands refer to the original development workspace; use this README
for the supported clean-package reproduction path. The validation search scripts
are retained for audit, not a request to resume tuning after the test was revealed.

## 5. Sources and substantive AI assistance

Codex assisted with architecture/optimizer proposals, code implementation,
experiment execution, debugging, validation search, packaging and report drafting.
The final model uses supplied training text only: no pretrained weights or external
training text. The student should review and understand the implementation and
report before submission. The report distinguishes observed evidence from hypotheses.

Sources: RoFormer (Su et al., arXiv:2104.09864); GLU Variants (Shazeer,
arXiv:2002.05202); Primer (So et al., arXiv:2109.08668); continuous cache
(Grave et al., arXiv:1612.04426); Keller Jordan's Muon (https://github.com/KellerJordan/Muon).
Muon adaptation retains `MUON_LICENSE.txt`. Local convolution uses one Primer idea,
not the complete architecture. Failed hashed-prefix experiments were inspired by
N-Grammer (arXiv:2207.06366) but used direct token hashes, not latent quantization.
Benchmark attribution is in `DATA_ATTRIBUTION.md`; the original classroom README
is preserved as `ORIGINAL_README.md`. No new license is asserted over teacher code.

## 6. Submission files

Upload this folder's contents to a repository and link a specific immutable commit.
Upload `MP1_model.zip` as the matching downloadable checkpoint bundle. The two
archives' SHA256 values are listed beside them in `SHA256SUMS.txt`. Report and data
must remain available with the code. The existing score issue is #19; use the
course site's workflow to add artifacts and preserve the original score JSON.
