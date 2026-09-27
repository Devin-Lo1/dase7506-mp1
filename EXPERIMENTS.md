# Student experiment: gated Transformer with rotary positions

## Method and hypothesis

The supplied GPT uses learned absolute positions and a GELU feed-forward layer.
The student predictor uses RoPE attention, a parameter-budget-matched SwiGLU
feed-forward layer, bias-free projections, residual-output initialization scaled
by sqrt(2 * depth), and residual dropout 0.1. Width increases from 128 to 192;
depth remains 4, vocabulary 2048 and context 256. Input/output weights are tied.
The hypothesis is that relative-position attention and gated features improve
prediction under the small CPU inference budget. This is an experimental
hypothesis, not a claim of optimality.

The learned-position ablation changes only the positional encoding setting.
It adds 49,152 learned position parameters. It has the same dropout, width,
SwiGLU, optimizer, training sampling seed and learning-rate horizon. Different
parameter initialization consumes different random numbers, so this is not an
identical-initialization paired experiment; only one seed is used.

## References and AI disclosure

- Su et al., RoFormer: https://arxiv.org/abs/2104.09864
- Shazeer, GLU Variants Improve Transformer: https://arxiv.org/abs/2002.05202

Substantive AI assistance: Codex proposed the method, implemented the student
model and checkpoint logging, ran experiments and drafted this documentation.
The student must understand and verify this implementation and report.
No external text or pretrained weights are used for learning. The references
supply algorithm ideas only. All model weights start from random initialization.

## Environment and reproduction

Python 3.12.14, PyTorch 2.7.1, NumPy 2.5.3, tokenizers 0.21.4.
Apple M5 MacBook Air, 16 GB RAM. CPU FP32, 4 threads, seed 17.
From `code/`, use the existing `.venv/bin/python` or create a Python 3.12 venv
and install `requirements.txt`. Commands require new, empty run directories.

```bash
python -m unittest discover -s tests -v
# Original baseline (already run):
python train.py --implementation model --device cpu --threads 4 --seed 17 --run-dir runs/baseline
# Proposed recipe, random initialization, with validation checkpoint selection:
python train.py --implementation student --config configs/student.json --device cpu --threads 4 --seed 17 --steps 2400 --eval-every 600 --run-dir runs/student-rope-s17
# Positional ablation at the same 1200-step budget and cosine horizon
# as checkpoint-step-1200.pt of the proposed run:
python train.py --implementation student --config configs/student_learned_positions.json --device cpu --threads 4 --seed 17 --steps 1200 --schedule-steps 2400 --eval-every 600 --run-dir runs/student-learned-s17
```

At 1,200 steps, both student variants and the original baseline have processed
9,830,400 training targets. The student variants have a 2,400-step cosine
horizon; the original baseline has a 1,200-step horizon. Thus the comparison
with the original baseline evaluates the whole changed recipe, not RoPE alone.
Compare student rotary and learned-position checkpoints **at step 1200** to
isolate the positional mechanism under their shared recipe. Comparing the final
2,400-step student to the original baseline also changes training budget;
do not attribute that entire improvement to architecture.

The 2,400-step run processes 19,660,800 targets. `best.pt` is selected using only
validation at steps 600, 1200, 1800 and 2400; `checkpoint.pt` is the last model.
All training/search cost counts, even if an earlier checkpoint is selected.
Neither optimizer state nor a resume facility is included; checkpoint files
are for direct inference. `metrics.json` records the training horizon, targets,
validation curve, selection, source hashes and timing.

## Evaluation protocol

The supplied `model.py`, `common.py`, `evaluate.py`, data and tokenizer are
preserved. Evaluation has no network access and no state across windows.
CPU resource comparisons use the same full validation split and settings;
these are development checks, not the final full-test resource certification.
Do not evaluate test until the method and checkpoint are frozen.

Inference needs only `student.py`, its PyTorch runtime and a selected checkpoint;
no retrieval index or external model asset is required. Dependencies are not
counted as model inference assets. Measure peak process RSS externally because
the supplied scorer reports GPU allocation fields as zero on CPU, not CPU RAM.

## Results

Pending completion. Baseline validation BPB: 2.0710874334. Baseline training:
316.4346 seconds; 1,088,256 parameters; 9,830,400 targets.

## Revised target and additional searches (23 September)

The requested target is below 1.4 BPB, with a final neural run of 1,200 updates.
The assignment does not fix batch size or total training duration. The next run
uses batch size 128, so 1,200 updates process 39,321,600 targets; its 300-update
checkpoint processes the same 9,830,400 targets as the original baseline.
The previous 2,400-update run is exploratory and is not the requested final run.

Completed small-model results on validation:

| Predictor | Neural updates / batch | Validation BPB |
|---|---:|---:|
| Original baseline | 1200 / 32 | 2.071087 |
| RoPE + SwiGLU student | 1200 / 32 | 1.762716 |
| RoPE + SwiGLU student | 2400 / 32 | 1.689102 |
| Student at 1200 + compact 5-gram counts | 1200 / 32 | 1.665701 |
| Student at 1200 + long counts, no cache | 1200 / 32 | 1.664706 |
| Student at 1200 + continuous cache, no counts | 1200 / 32 | 1.704033 |
| Student at 1200 + long counts + continuous cache | 1200 / 32 | 1.617070 |

These are validation selections, not held-out test results or proof of optimality.
The cache reference is Grave et al., https://arxiv.org/abs/1612.04426 . Unlike a
cross-document cache, this implementation resets for every independent 256-token
window. At prediction position t, it uses hidden-state keys h[j] only when j<t,
with values ids[j+1], all already observed at t. No cross-window cache is used.

The count model is fitted only to training text (3,613,343 tokens), with lower
orders based on continuation counts and interpolated absolute discounting.
Singleton higher-order transitions are pruned; their mass backs off to lower
orders. Contexts up to order 5 use exact base-2048 keys; orders 6 through 12 use
a fixed polynomial hash modulo 2^63. Hash collisions are theoretically possible,
so the long-context representation is approximate. The saved statistics occupy
30,336,734 tensor bytes. All model assets must still satisfy the 64 MiB limit.

Count construction reads training text in addition to neural training: this is
additional training/search cost, not free computation. Report it separately and
compare count-enabled and count-disabled models as an inference-mechanism
ablation. Neural-only checkpoint comparisons remain matched in processed targets.
The extension builder makes another pass over the training token stream.

Searches stream validation probabilities to accumulate candidate losses. They
never supply targets to the predictor and never export per-target answers as
model assets. Only global interpolation weights, temperature, and cache sharpness
are selected. `search.json` retains the candidate aggregate scores.

```bash
python build_ngram.py --output runs/ngram/train-kn5-pruned.pt
python extend_ngram.py
python tune_hybrid.py --neural runs/student-rope-s17/checkpoint-step-1200.pt --output-dir runs/hybrid-small-1200
python tune_cache.py --neural runs/student-rope-s17/checkpoint-step-1200.pt --output-dir runs/hybrid-cache-small-1200
python train.py --implementation student --config configs/student_large.json --device cpu --threads 4 --seed 17 --steps 1200 --batch-size 128 --learning-rate 0.002 --warmup-steps 60 --decay-matrices-only --eval-every 300 --run-dir runs/student-large-b128-s17
```

The earlier proposed learned-position training ablation has not been run; the
new target prompted prioritizing larger models and count/cache mechanism ablations.
The first unpruned-context count export was 88,674,222 tensor bytes and was
rejected as too large. Its construction time (7.117 seconds) remains part of
search cost; the compact builder removes contexts with no retained transitions.

## Apple GPU training path

A direct CPU batch-128 run was interrupted before the first 100-update log
because of memory pressure and swapping. No checkpoint from that attempt is
used. Its elapsed search cost was approximately 450 seconds; the exact number
of completed updates was not logged. `runs/student-large-b128-s17/aborted.json`
records the unsuccessful attempt.

The student trainer now supports MPS **only for training**, without changing
`common.py` or `evaluate.py`. PyTorch documents the MPS training backend at
https://docs.pytorch.org/docs/stable/notes/mps.html . Validation creates a
separate CPU model copy and calls the unchanged scorer in FP32. The original
scorer's float64 reductions are not executed on MPS. Final checkpoint evaluation
uses the original CPU command and does not require an Apple GPU.

```bash
python train.py --implementation student --config configs/student_large.json --device mps --precision fp32 --threads 4 --seed 17 --steps 1200 --batch-size 64 --accumulation-steps 2 --learning-rate 0.002 --warmup-steps 60 --decay-matrices-only --log-every 25 --eval-every 300 --run-dir runs/student-large-mps-b128-s17
```

Each loss is divided by the number of accumulation microbatches; gradients are
clipped and AdamW is updated once per complete effective batch of 128 examples.
The run has 1,200 optimizer updates, 2,400 microbatches, and 39,321,600 targets.
Dropout and device arithmetic mean this is not bitwise identical to CPU batch-128
training. CPU checkpoint scoring, not identical cross-device training, is the
reproduction requirement. The original 1,200-step CPU baseline remains saved.

A full-validation resource screen with the large architecture and count/cache
components (random neural weights, so not a quality result) measured 22.9463 s
vs baseline 5.8945 s, ratio 3.893; peak process RSS 2.498 GiB; unique inference
model tensors 55.465 MiB. This is a preliminary screen only; measure the final
trained predictor again, and use full test for final certification after freezing.

An additional strictly causal suffix-copy mechanism improved the small-model
combination from 1.617070 to 1.605641 validation BPB. It finds the longest matching
suffix (up to six tokens) among previously observed prefixes inside the current
window. When at least two tokens match, it mixes in the empirical continuation
distribution with validation-selected weight 0.3. The observed continuation is
always at an index <= the current prediction position. Repeated-sequence tests
check future-token invariance, normalization, batch independence and reset.

```bash
python tune_copy.py --checkpoint runs/hybrid-cache-small-1200/checkpoint.pt --output-dir runs/hybrid-copy-small-1200
```

The unmodified scorer independently reproduced the pre-copy combination at
1.6170696065 BPB (vs search 1.6170696094). Its 25.5057-second timing was taken
while training was active and must not be used as the final resource comparison.
The main reported search/ablation comparisons use the same validation split.

## Completed large AdamW run and further candidates

The 1,200-update, effective-batch-128 MPS FP32 AdamW run completed in
2,492.761 training seconds (39,321,600 targets). CPU validation BPB at updates
300/600/900/1200 was 1.758519 / 1.653898 / 1.619228 / 1.609495.
After validation selection, the count/continuous-cache combination reached
1.521223 (temperature 1.1, count weight 0.2, cache weight 0.05, sharpness 20).
Suffix copying with weight 0.3 reduced it to 1.508591. This remains above 1.4.

The next full run uses stronger dropout (residual 0.15, embedding 0.1,
attention 0.05), weight decay 0.3, and Muon for hidden matrix weights with
auxiliary AdamW for embeddings and scalar/vector parameters. It is again random
initialization, 1,200 updates and effective batch 128. The auxiliary AdamW uses
betas (0.9, 0.95). Muon learning rate is 10x the scheduled auxiliary learning rate;
its decay is divided by 10 to keep the decoupled decay factor comparable.
Muon source: https://github.com/KellerJordan/Muon/blob/master/muon.py .
`muon_local.py` adapts and batches its single-device algorithm; the upstream
MIT notice is preserved in `MUON_LICENSE.txt`. This changes both optimization
and regularization, so it is a whole-recipe comparison, not a Muon-only ablation.

```bash
python train.py --implementation student --config configs/student_regularized.json --device mps --precision bf16 --threads 4 --seed 17 --steps 1200 --batch-size 64 --accumulation-steps 2 --learning-rate 0.002 --warmup-steps 60 --optimizer muon --weight-decay 0.3 --log-every 25 --eval-every 300 --run-dir runs/student-muon-reg-b128-s17
```

MPS BF16 autocast was tested on this machine. Model/optimizer master weights
remain FP32 and all recorded validation uses CPU FP32. The two-update Muon
smoke test adds 16,384 training targets and 5.714 training seconds to search cost.
A stricter composite-model normalization check exposed FP32 residual errors
around 1.6e-6; the predictor now subtracts logsumexp before returning probabilities.
This preserves the intended mixture to floating precision and satisfies the
normalization contract; the fixed scorer is unchanged.

## Experimental compact nearest-neighbor augmentation

Reference: Khandelwal et al., Generalization through Memorization: Nearest
Neighbor Language Models, https://arxiv.org/abs/1911.00172 . This project uses
its own randomly trained model, not any externally pretrained model.

`build_knn.py` samples 1,000,000 training positions with seed 17, computes the
frozen model's causal hidden state at each selected position, and stores the
observed training next-token ID. A train-fitted PCA projects hidden states to
64 dimensions. Faiss IVF2048/PQ8x8 compresses the index; query settings and
interpolation are selected on validation. Index construction is additional
training-derived processing and must be included in search cost.

`development/training-features.f16` is temporary build input, not an inference
asset. A final bundle may contain only the checkpoint, runtime source, projection,
compressed index and training-token values listed in the metadata. The 64 MiB
limit applies to their complete uncompressed total. The kNN variant retains only
the <=5-gram count tables to make room. Retrieval adds computation and may be
rejected if the trained predictor exceeds the CPU time budget.

```bash
python -m pip install -r requirements-student.txt
python configure_faiss_macos.py  # macOS only; a no-op on other OSes
python build_knn.py --checkpoint runs/student-large-mps-b128-s17/checkpoint-step-1200.pt --output-dir assets/knn-adam-large-1200
python tune_knn.py --base-checkpoint runs/hybrid-copy-large-1200/checkpoint.pt --asset-dir assets/knn-adam-large-1200 --output-dir runs/knn-adam-large-1200
```

On this Mac the Faiss and PyTorch wheels initially loaded separate OpenMP
runtimes and aborted. `configure_faiss_macos.py` makes Faiss share PyTorch's
unchanged libomp, retaining the original Faiss library as a backup inside the
virtual environment. An exact nearest-neighbor numerical check passed after
the change. This adjustment must be repeated after reinstalling the Faiss wheel.


## Muon completion and continued 1,200-update search (24 September)

The effective-batch-128 Muon/regularization recipe completed all 1,200 updates
from random initialization. CPU FP32 validation BPB at updates 300/600/900/1200
was 1.765520 / 1.620810 / 1.564767 / **1.542515**. It processed 39,321,600
training targets and spent 2,192.592 seconds training (2,311.146 seconds total).
Count/continuous-cache interpolation reduced validation BPB to 1.495854;
constant suffix-copy weight 0.2 reduced it further to **1.487592**.
These are validation selection results, not a leaderboard test score.

A 50/50 average of this run's step-900 and step-1200 weights was worse after
count/cache interpolation (1.498920 versus 1.495854), so that average is rejected.
Its ancestry remains 1,200 updates, not 2,100 independent training updates.
The earlier Adam model's compact kNN variant reached validation 1.504584,
with 59.577 MiB of checkpoint/index assets; its small gain did not outperform
this new Muon candidate. The kNN model has not been selected for submission.

A new random-initialization Muon run uses the same configuration, seed, optimizer
and schedule but effective batch 256 (microbatch 64, accumulation 4). It still
performs 1,200 updates but processes 78,643,200 training targets. This is allowed
by the assignment's unrestricted training budget and is not an equal-target
comparison with the baseline. It is recorded as pending until metrics.json
confirms completion.

```bash
python train.py --implementation student --config configs/student_regularized.json --device mps --precision bf16 --threads 4 --seed 17 --steps 1200 --batch-size 64 --accumulation-steps 4 --learning-rate 0.002 --warmup-steps 60 --optimizer muon --weight-decay 0.3 --log-every 25 --eval-every 300 --run-dir runs/student-muon-reg-b256-s17
python finish_candidate.py --training-dir runs/student-muon-reg-b256-s17 --output-dir runs/pipeline-muon-b256 --wait --measure-resources
```

`finish_candidate.py` waits for completed training, then runs count/cache,
suffix-length copy and adaptive-count validation selection, verifies the chosen
checkpoint with the unchanged scorer, and optionally measures CPU time/RAM/assets.
It never evaluates test. Run its resource stage without concurrent training or
other scoring processes. Every stage saves its command, exit status and full log.

`tune_order_copy.py` selects three global copy weights for matches of length 2,
3, and at least 4; predictions depend only on the observed prefix. The grouping
and candidate weight grid are fixed before examining this experiment's results.
The CPU suffix implementation was checked against its Torch reference, including
match lengths, short inputs and length-256 inputs. Original five contract tests
pass, and ten protected release files (data, tokenizer, scorer, baseline and
tests) match their original hashes.

`refine_count_asset.py` tests retaining singleton trigram continuation counts.
It derives these counts solely from unique training four-grams and losslessly
packs integer context IDs/pointers. Other orders retain the original pruning.
The resulting count tensors occupy 35,163,000 bytes and took 24.209 seconds to
refine, with an additional scan of the training split. Performance is pending;
asset packing does not change the numeric meaning of the stored integers.


### Follow-up validation outcomes and implementation checks

For the Muon batch-128 checkpoint, length-dependent copying reached 1.487183.
Adaptive count mixing (temperature 1.1, low weight 0.1, high 0.5, confidence
power 2) reached 1.482912. Retaining singleton trigram continuation counts
improved this to **1.477186435** with the unchanged CPU FP32 scorer.
The full-trigram checkpoint is about 60.11 MiB before adding runtime source.
No test score has been measured and no final resource certification is claimed.
The raw scorer times during concurrent MPS training (121–174 seconds) are
contended measurements and must not be used as budget certification.

Count inference now expands the backoff recurrence from high to low order.
It accumulates sparse contributions and applies the remaining unigram mass once,
which avoids repeatedly multiplying the entire dense vocabulary distribution.
The original recurrence remains as `_count_probabilities_reference`. Random,
repeated and real validation prefixes, including lengths 1 and 256, agreed to a
maximum observed probability difference below 9e-8. Full fixed-scoring results
also agree with selection likelihoods within 1e-7 BPB. A concurrent microbenchmark
showed 0.400 versus 0.285 seconds for one batch; this is not a formal speed claim.

Reproduce the singleton-count ablation with:

```bash
python refine_count_asset.py --output runs/ngram/train-kn12-full-trigram.pt
python replace_count_asset.py --checkpoint runs/hybrid-order-copy-muon-1200/checkpoint.pt --counts runs/ngram/train-kn12-full-trigram.pt --output runs/hybrid-full-trigram-muon-1200/checkpoint.pt
python tune_adaptive.py --checkpoint runs/hybrid-full-trigram-muon-1200/checkpoint.pt --output-dir runs/hybrid-full-trigram-adaptive-muon-1200
python evaluate.py --checkpoint runs/hybrid-full-trigram-adaptive-muon-1200/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
```

A further architecture candidate, `student_local.py`, adds causal length-three
depthwise convolutions after Q/K/V projection, inspired by Primer:
https://arxiv.org/abs/2109.08668 . It keeps SwiGLU and identity-initializes the
convolutions, so it is not the complete Primer recipe. CPU evaluation computes
the same convolution with explicit causal shifts to avoid slow grouped-convolution
dispatch; a direct numerical comparison passed. All eleven base/local contract
tests pass, with nonidentity random convolution weights to exercise causality.
This architecture is currently prepared, not a trained improvement.

`collect_experiments.py` inventories completed runs and validation searches.
Before completion of the batch-256 run, completed gradient training totaled
108,232,704 targets and 5,955.056 seconds, excluding the aborted CPU run and
preprocessing/index construction. Its output explicitly separates pending runs,
validation selection and independently verified validation scores. Count-fitting
corpus size is not an optimizer-update count: its algorithms traverse multiple
n-gram orders. kNN index construction also performs a separate train-only forward
scan; these costs remain in their own metadata and earlier experiment records.


### Scheduled local-convolution experiment

The local model's two-update MPS BF16 smoke run completed with finite losses and
CPU FP32 validation. This is only an execution check: 4,096 training targets,
9.693 training seconds (including GPU initialization), 125.961 process seconds
under concurrent training load. Its 3.41688 BPB is not a useful trained-model
comparison. The local model has 6,973,952 parameters.

A bounded follow-up queue waits for the batch-256 experiment and its CPU resource
measurement to finish. If that earlier candidate is still above 1.4 validation
BPB or fails resource limits, it profiles the local architecture with a packed
<=5-gram count asset. It launches at most one further random-initialization run,
and only if the prototype satisfies all resource limits. The follow-up uses
1,200 updates, effective batch 256, peak auxiliary learning rate 0.003, Muon,
and the same dropout recipe. Raising learning rate together with changing the
architecture makes this a recipe comparison, not an isolated convolution ablation.
The original batch-256 and batch-128 runs remain available as controls.

```bash
python run_local_candidate.py --previous-pipeline runs/pipeline-muon-b256/pipeline.json --output-dir runs/local-followup-queue
```

All commands and logs are written to `runs/local-followup-queue/queue.json` and
its sibling logs. The queue never evaluates test or starts unbounded extra trials.
Its resource prototype is explicitly marked and must never be reported as an
accuracy candidate. The final method is still unfrozen.


## Completed batch-256 Muon run and resource-wrapper recovery

The batch-256 run completed 1,200 updates / 78,643,200 targets. Training took
4,236.240 seconds, total process time 4,334.952 seconds. CPU FP32 validation at
300/600/900/1200 was 1.679858 / 1.572919 / 1.539142 / **1.525938**.
Its count/cache, length-dependent copy and adaptive-count stages reached
1.475024 / 1.467066 / **1.464080025**, respectively. The last number was
independently verified by the fixed scorer. The gains from doubling effective
batch size are modest relative to the extra training cost.

The unattended selection pipeline completed, but resource measurement failed:
macOS `/usr/bin/time -l` could not access `sysctl kern.clockrate` in the sandbox.
The baseline scorer itself succeeded. The follow-up queue therefore stopped
before launching new training. No numerical/resource success is inferred from
that failed wrapper call.

`measure_validation.py` now launches the unchanged scorer in a fresh Python
process and records that process's own peak RSS using `resource.getrusage` at
exit. Scoring time still comes from the fixed scorer. RSS is bytes on macOS and
converted from KiB on Linux. This avoids the restricted system-wide clock query.
The failed log is retained; new alternating measurements are recorded separately
under `runs/pipeline-muon-b256/resources-retry`.


Resource retry succeeded for the batch-256 adaptive hybrid: median CPU ratio
4.135195, peak RSS 2.594269 GiB, complete listed inference files 55.528413 MiB.
Three alternating baseline/candidate runs were used with no model training in
progress. The original failed wrapper stage remains in pipeline.json and is
superseded by a recorded successful resource retry.

The first local-convolution prototype measured 5.158923x baseline time and was
rejected before full training (2.600830 GiB, 48.331734 MiB). The prototype's score
is irrelevant: it contains only two-update smoke weights. CPU inference now uses
in-place addcmul for the causal shifts when gradients are disabled, eliminating
padded temporary tensors. The differentiable CPU path and MPS convolution are
unchanged. A direct full-prediction comparison of fused/non-fused CPU paths
passed. A new isolated three-repeat resource profile is running under
`runs/local-followup-fused`; the earlier failed prototype measurement is retained.


## Completed local-convolution recipe (25 September)

The optimized local-convolution candidate passed its prototype guard and completed
1,200 updates, effective batch 256, 78,643,200 training targets, peak auxiliary
learning rate 0.003. Training took 4,299.904 seconds; total trainer process time
is recorded in metrics.json and the queue log. Validation BPB at updates
300/600/900/1200 was 1.645713 / 1.537799 / 1.507527 / **1.496975**.
The final count/cache/copy mixture achieved **1.447069544** with the unchanged
CPU FP32 scorer. Three alternating resource measurements gave **4.283643x**
baseline scoring time, **2.591232 GiB** peak RSS, and **48.333113 MiB** assets.
All development resource limits pass. Test remains unevaluated.

The next prepared recipe uses width 384, five local-attention blocks, six heads,
residual dropout 0.2, embedding dropout 0.1 and attention dropout 0.05, with
9,659,520 trainable parameters. The trainer now exposes `--min-lr-ratio`;
its default 0.1 preserves the previous schedule exactly. A floor of zero makes
the last update's multiplier approximately 1.7e-6 of peak, instead of 0.1.
This is a proposed optimization/regularization experiment, not an observed gain.
The wider model passed causal-prefix and normalization checks before profiling.


## Same-lineage averaging, full counts, and wider follow-ups

The local batch-256 model's 900/1200 checkpoints averaged 25%/75% scored
1.445907273 validation BPB. 50%/50% was worse (1.44724059).
Replacing counts with the train-derived full singleton-trigram KN12 asset scored
**1.4449589834512648** using the unchanged CPU FP32 scorer, checkpoint SHA256
9bf688e75f04999f90de6475db1571783445d7dcaa0d327f152f6ea278f97f96.
This checkpoint is 60.1813 MiB; its final resource measurement is pending.
No extra gradient targets are introduced by averaging; ancestry remains 1200 updates.

The 384-wide, five-layer model passed a prototype resource check at 4.794162x,
2.56514 GiB, 58.56742 MiB. It trains with microbatch 64 and accumulation 8,
effective batch 512, peak auxiliary LR .004 and cosine floor zero, for 1200 updates
(157,286,400 gradient targets). Validation at 300/600/900 is
1.5761510553 / 1.5036658317 / 1.4916423383. Its step-75 equal-target control scores
2.02612880821 versus the original baseline's 2.0710874334 after 9,830,400 targets.
Optimizer, architecture, batch and schedule differ; this is a whole-recipe comparison.
The concurrent scoring duration is not a valid standalone resource certification.

A conditional 448-wide, five-layer run waits until the wider pipeline finishes.
It first profiles resources and only trains if all limits pass. It uses a losslessly
integer-packed KN5 pruned count asset (13,537,206 tensor bytes) to fit the size limit.
The pack operation verifies every integer value, and performs no further data scan.

## Learned prefix hash augmentation: implementation checks

Added student_hash.py, inspired by neural n-gram augmentation (N-Grammer,
https://arxiv.org/abs/2207.06366), not a reproduction: direct token-ID hashes replace
latent vector quantization. Two random-initialized learned tables represent causal
bigram/trigram prefixes, masked at the start of each independent window, projected
into the token input. There are no pretrained weights or validation-derived entries.
The proposed 384-wide four-layer model has 10,006,784 parameters and hash tables
of 32,768 buckets x 32 dimensions. It has not been trained as an accuracy candidate.

The first MPS smoke check failed: integer remainder produced CPU/MPS hash differences
of one on 11/34 entries. This probe consumed two updates / 64 targets; elapsed
training time was not persisted before the assertion, and no checkpoint was exported.
The failure is retained in runs/hash-mps-smoke-failed/aborted.json. Requiring power-of-two
bucket counts and applying bitwise AND fixed the mismatch. The first successful
probe consumed another 64 targets in 1.489818 seconds. A subsequent expanded check
covers random token IDs at full context length. All 19 CPU contract tests passed.
Neither smoke probe provides evidence of improved BPB.


## Completed wide model and rejected XL prototype

The 384-wide five-layer recipe finished 1,200 updates in 8,948.829943 training
seconds. Raw validation BPB is 1.488248128; the validated adaptive mixture is
**1.4243763294422123** (grid prediction 1.4243761538350534). Resource measurement
passed: 4.772536x baseline, 2.574112 GiB, 58.568878 MiB. The 448-wide five-layer
prototype failed at 5.337900x despite passing RAM and size (2.566010 GiB,
63.056905 MiB), so no complete XL training was run.

The learned-hash prototype passed at 4.240519x, 2.572815 GiB, 59.892593 MiB.
Its 900-update validation of 1.669594 shows substantial overfitting relative to
the wide model despite lower training loss. It is not selected.
A new validation-only coordinate search (tune_joint.py) adjusts existing global
temperature, count discount/interpolation, cache and suffix-copy weights together.
Only global settings and aggregate losses are saved; per-target probabilities
are temporary search RAM, never inference assets. The independent fixed scorer
will verify any selected improvement.


A further bounded random-initialized 1200-update recipe is queued after current
selection and resource profiling finish: same 384x5 architecture, batch 512, LR
.004, zero cosine floor, but residual/embedding/attention dropout .3/.2/.1 and
auxiliary weight decay .6 (Muon matrix decay .06). This tests the hypothesis
that stronger regularization reduces the observed training-validation gap. It is
a hypothesis, not an established improvement. No initial weights are transferred.


Joint global calibration completed in 125.827470 seconds. Two coordinate sweeps
selected cache weight .025 (previously .05) and suffix-copy weight .25 for
length >=4 (previously .2); other settings remained unchanged. Fixed CPU FP32
validation independently reproduced **1.4217038295231355 BPB** versus search
1.4217036444342634. Checkpoint SHA256:
c3546391a7b8c2636de9fc0555faf3e614f454b2871ccfc5f7102770cb559069.
The concurrent 72.829-second scorer timing is not resource certification.
The new combination awaits isolated resource measurement after queued training.
All ten protected benchmark files were verified unchanged against PACKAGE_MANIFEST.


## Final frozen full-test evaluation

The user requested closure and final testing. The not-yet-started stronger-regularization
queue was cancelled; its prototype never became a trained candidate. Completed
hash and same-lineage averaging trials did not beat the joint-calibrated wide model.
The validation-only resource check selected the 1.4217038295 candidate and froze
checkpoint c3546391a7b8c2636de9fc0555faf3e614f454b2871ccfc5f7102770cb559069
and runtime sources before any test score. Freeze timestamp: 2026-09-25T19:25:28.870873+00:00.

Three full-test CPU FP32 evaluations reproduced **1.4381909205827006 BPB**,
428,405 scored targets and 1,292,013 UTF-8 bytes. Baseline test BPB was
2.1012651958438786. Three alternating repetitions gave 4.672703x median time,
2.580750 GiB peak RSS and 58.568878 MiB inference assets. All limits passed.
The <1.4 target was not achieved. No further predictor tuning or selection used
the test results. Frozen files and benchmark source hashes were reverified.
Validation-only component removal ablations are diagnostic and do not alter the
frozen predictor or its settings.


Final frozen-model validation-only component ablations (no retuning):

| 完整最终模型 | 1.421703830 |
| 移除训练集 n-gram 统计 | 1.436440911 |
| 移除连续缓存 | 1.424676534 |
| 移除后缀复制 | 1.426595268 |
| 仅保留校准后的神经模型 | 1.448975318 |

Ablation scoring took 135.48347620799905 seconds; no additional gradient targets.
These diagnostics were not used to change the frozen predictor.
