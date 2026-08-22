# VGGT-MLX Public Release Roadmap

> Implementation guide for turning the working fp32 port into a discoverable,
> reproducible public release. Execute cards in order. Each card is complete only
> when its acceptance gate passes and its evidence is committed.

## Release objective

Ship the first credible, numerically validated, native MLX implementation of
VGGT-1B for Apple Silicon.

The primary claim is correctness:

> VGGT-1B rebuilt in MLX, running locally on Apple Silicon, and validated
> layer-by-layer against the official PyTorch implementation.

A performance claim may replace it as the headline only after a controlled,
same-machine comparison establishes a reproducible advantage over PyTorch MPS.

## Competitive position

| Project | Relationship | Strength to match | VGGT-MLX opportunity |
|---|---|---|---|
| `jmanhype/vggt-mps` | Direct VGGT alternative | Easy CLI, MPS support, sparse-attention option | Native MLX, stricter numerical evidence, measured same-machine comparison |
| `Aedelon/mlx-mast3r` | Adjacent MLX 3D project | PyPI, automatic weights, fp16, custom kernels, strong benchmark story | Apply the same release quality to VGGT specifically |
| Upstream VGGT | Reference implementation | Complete feature set and official weights | Local Apple-Silicon inference without PyTorch at runtime |

Do not describe MASt3R as a VGGT port or claim that VGGT-MLX is faster than it;
the models and workloads are different.

## Non-negotiable release rules

1. **Preserve the fp32 reference.** Existing CPU/fp32 parity gates must remain
   unchanged while performance work proceeds.
2. **Benchmark identical work.** MLX and PyTorch MPS must receive the same
   preprocessed arrays, checkpoint family, output heads, view count, and image
   dimensions.
3. **Synchronize lazy execution.** Evaluate every MLX output before stopping a
   timer. Synchronize MPS before and after each measured trial.
4. **Keep raw evidence.** Every public number must be generated from a committed
   machine-readable result containing raw samples and environment metadata.
5. **Never hide invalid runs.** Retain failures and thermally unstable runs with
   an exclusion reason; do not silently replace them.
6. **Do not weaken parity to gain speed.** fp16 and bf16 are separately labeled
   operating modes with task-level quality reports, not replacements for fp32.
7. **Do not redistribute weights until approved.** Code, the original
   non-commercial checkpoint, the commercial checkpoint, and converted weights
   have related but distinct obligations. Complete Card R0.2 first.
8. **Use exact, bounded language.** Prefer “measured on an M4 with 16 GB” over
   “fast,” and “tracking head not implemented” over “full VGGT.”

## Target user experience

The release should reduce first use to:

```bash
pip install vggt-mlx
vggt-mlx setup
vggt-mlx reconstruct photos/ --output output/
```

`setup` may download approved converted weights or perform a guided local
conversion, depending on the outcome of the license review. Inference must not
require PyTorch.

---

## R0 — Freeze the release baseline

### Card R0.1 — Record the current golden revision

**Objective:** Establish an immutable correctness and output baseline before
optimization.

**Deliverables:**

- `release/baseline.json` containing the git revision, Python/macOS/MLX versions,
  machine model, unified memory, test result, current parity values, checkpoint
  source, and converted-weight checksum;
- regenerated one-view and three-view demo outputs for visual comparison;
- a tag or documented commit identifying the fp32 baseline.

**Acceptance gate:**

- `pytest -q` passes with converted weights available;
- all existing fp32 parity thresholds pass unchanged;
- the baseline file validates and contains no placeholder values;
- the repository is clean after regenerating evidence.

### Card R0.2 — Resolve distribution and attribution

**Objective:** Determine exactly what may be published for each checkpoint.

**Deliverables:**

- `docs/DISTRIBUTION.md` distinguishing:
  - original VGGT source code;
  - `facebook/VGGT-1B`;
  - `facebook/VGGT-1B-Commercial`;
  - locally converted MLX derivatives;
  - sample images and generated outputs;
- required notices, attribution, paper citation, acceptable-use language, and
  commercial/non-commercial restrictions;
- an explicit decision for each artifact: publish, conversion-only, gated, or
  excluded;
- upstream source and checkpoint revisions pinned in conversion metadata.

**Acceptance gate:**

- README wording agrees with the current upstream license and model cards;
- no model-weight upload begins without an explicit “publish” decision for that
  exact derivative;
- the repository contains no restricted or accidentally tracked checkpoint.

### Card R0.3 — Define versioned public result schemas

**Objective:** Make every performance and parity claim traceable.

**Preferred home:** implement generic schema and metrics in
`mlx-vision-bench`; keep a thin adapter and result files here.

**Required benchmark fields:**

- schema version, run ID, UTC timestamp, git revision, dirty state;
- hardware, memory, macOS, Python, MLX, PyTorch, power state;
- framework, precision, checkpoint revision and checksum;
- input checksums, views, tensor dimensions, enabled outputs;
- warmup count, trial count, raw milliseconds, median, IQR, minimum;
- peak-memory reading and measurement API;
- thermal stability flag, inclusion status, and exclusion reason.

**Required parity metrics:**

- maximum and mean absolute error;
- relative Frobenius error;
- cosine similarity;
- camera rotation geodesic error in degrees;
- camera translation direction error in degrees;
- first failing named tap and versioned tolerance policy.

**Acceptance gate:**

- schema validation rejects missing samples, unknown policy versions, and an
  excluded run without a reason;
- analytical unit tests cover every numerical metric;
- README tables can be regenerated from result files rather than edited by hand.

---

## R1 — Reproducible MLX versus PyTorch-MPS benchmark

### Card R1.1 — Add a PyTorch-MPS reference runner

**Objective:** Run official VGGT on MPS using precisely the same input arrays and
requested output heads as the MLX adapter.

**Implementation requirements:**

- preprocess once with the shared NumPy/Pillow path;
- feed identical fp32 arrays to both frameworks;
- pin the upstream source and checkpoint revisions;
- isolate model loading, preprocessing, inference, and export timings;
- use `torch.mps.synchronize()` around each measured trial;
- make tracking exclusion explicit so comparisons cover equivalent outputs.

**Acceptance gate:**

- MLX and MPS adapters report identical input checksums and shapes;
- both return the agreed named output set;
- the MPS output passes a documented reference comparison before it is timed.

### Card R1.2 — Implement the benchmark runner

**Objective:** Produce resumable, reproducible timing cells.

**Default matrix:**

- frameworks: MLX and PyTorch MPS;
- precision: fp32 initially;
- views: 1, 2, 3, and 4;
- workload: model forward only, with separately reported preprocessing and
  point-cloud export;
- warmups: at least 5;
- measured trials: at least 10;
- fixed approved image bundle.

Record wall time per scene and per frame. Do not use per-frame time alone for a
multi-view transformer because scaling is not necessarily linear.

**Acceptance gate:**

- rerunning skips completed valid cells unless `--replace` is explicit;
- MLX timing forces evaluation of all requested outputs;
- raw samples reproduce the reported median and IQR;
- a run with `max(sample) / min(sample) > 1.2` is flagged for review;
- interrupted sweeps resume without overwriting valid results.

### Card R1.3 — Collect and publish the fp32 comparison

**Objective:** Establish whether native MLX wins on the current implementation.

**Acceptance gate:**

- all cells run on AC power with the machine and environment recorded;
- included cells are stable and schema-valid;
- README table is generated from committed result JSON;
- any speedup is calculated from same-machine medians;
- CUDA results, if included, are clearly informational and never presented as a
  controlled speedup comparison across different hardware.

**Decision:**

- If MLX wins materially, use the measured range as the launch hook.
- If performance is comparable or slower, keep numerical validation as the hook
  and use the profile from R1.4 to drive optimization.

### Card R1.4 — Profile the slow path

**Objective:** Identify optimization targets rather than guessing.

Measure or isolate:

- DINO patch backbone;
- frame and global attention;
- RoPE;
- camera head;
- DPT resize and fusion blocks;
- depth unprojection and PLY export;
- evaluation boundaries and avoidable host transfers.

**Acceptance gate:**

- the top three costs are supported by measurements;
- every proposed optimization names the affected parity tap;
- no custom Metal kernel is started without a measured bottleneck and a
  standalone PyTorch/NumPy reference test.

---

## R2 — Reduced precision and performance pass

### Card R2.1 — Add an explicit precision policy

**Objective:** Support `fp32`, `fp16`, and `bf16` without implicit casting.

**Implementation requirements:**

- expose precision through model loading and the CLI;
- document which parameters and operations remain fp32;
- keep normalization, softmax, pose conversion, and other sensitive operations
  in fp32 where required by evidence;
- generate reduced-precision weights deterministically or cast safely at load;
- report memory separately for each mode.

**Acceptance gate:**

- fp32 outputs and existing parity results remain unchanged;
- fp16/bf16 outputs contain no NaN or infinity for 1–4 approved views;
- strict loading succeeds for every supported weight format.

### Card R2.2 — Establish reduced-precision quality gates

**Objective:** Quantify task drift against MLX fp32.

Compare depth, confidence, pose, extrinsics, intrinsics, and world points using
the metrics defined in R0.3. Include at least the one-view and three-view oracle
scenes plus one visually distinct held-out scene.

**Acceptance gate:**

- full metric reports are committed for both fp16 and bf16;
- a precision is labeled supported only if it stays finite and meets the
  versioned policy;
- visually inspect side-by-side depth and registered point clouds;
- a failing mode remains experimental rather than receiving looser thresholds.

### Card R2.3 — Apply measured MLX optimizations

**Candidates, only when supported by R1.4:**

- reduce unnecessary `mx.eval` and NumPy transfer boundaries;
- compile stable graph regions with `mx.compile`;
- fuse or replace slow RoPE/resize operations;
- add custom Metal kernels for proven hotspots;
- cache invariant positional data;
- reduce retained intermediate activations;
- make output heads selectable when users do not need all products.

**Acceptance gate for each optimization:**

- standalone numerical test passes;
- full fp32 parity passes unchanged;
- reduced-precision quality gates still pass;
- at least three controlled before/after runs demonstrate a repeatable gain;
- code retains a clear fallback when a custom kernel is unavailable.

### Card R2.4 — Publish the optimized benchmark matrix

Repeat R1.2 for supported precisions. The headline number must identify the
machine, precision, resolution, views, outputs, baseline, and statistic.

Acceptable example:

> On an M4/16 GB, VGGT-MLX fp16 reconstructed the approved two-view scene in
> X ms median across 10 trials, Y× the equivalent PyTorch-MPS run.

Unacceptable examples include “real time,” “fastest,” or “X× faster” without a
bounded workload and reproducible evidence.

---

## R3 — Installation and product surface

### Card R3.1 — Build the `vggt-mlx` CLI

**Commands:**

```text
vggt-mlx setup
vggt-mlx reconstruct INPUT... --output DIR --precision fp16
vggt-mlx inspect
vggt-mlx benchmark
```

`setup` verifies the platform, disk space, checkpoint source, checksum, and
license acknowledgement. `inspect` reports versions, available memory, weight
status, and supported precision. `reconstruct` accepts files, globs, or a scene
directory and produces depth PNGs, cameras, and PLY output.

**Acceptance gate:**

- a fresh virtual environment can reach a reconstruction by following only
  `vggt-mlx --help`;
- errors for Intel Macs, missing weights, insufficient disk, unrelated images,
  and excessive view counts are actionable;
- inference does not import PyTorch;
- existing `demo.py` either delegates to the CLI or is clearly deprecated.

### Card R3.2 — Implement approved weight acquisition

**Path A — redistribution approved:**

- publish safetensors, config, conversion manifest, original revision, source
  checksum, converted checksum, license, and model card;
- make download resumable and checksum-verified;
- never silently switch between original and commercial checkpoint families.

**Path B — conversion only:**

- make `vggt-mlx setup` create an isolated temporary upstream environment,
  download the user-selected official checkpoint, convert it, verify strict
  loading, and clean temporary files safely;
- support reruns without redownloading valid cached artifacts.

**Acceptance gate:**

- test from a clean user account or clean virtual environment;
- interrupted download/conversion resumes or fails safely;
- a corrupt checkpoint is rejected before inference;
- no access token or checkpoint is committed or logged.

### Card R3.3 — Package and publish to PyPI

**Deliverables:**

- complete project metadata, classifiers, URLs, license expression, and package
  data;
- version sourced from one location;
- built wheel and source distribution;
- installation smoke test from the built wheel;
- TestPyPI rehearsal before production publication.

**Acceptance gate:**

- wheel install works in a clean environment;
- `vggt-mlx --help` and a lightweight no-weight smoke test pass;
- package does not accidentally contain weights, fixtures, outputs, or secrets;
- release tag and package version agree.

---

## R4 — Public evidence and documentation

### Card R4.1 — Create the launch demo

Produce a 10–20 second, silent-loop-friendly video or GIF showing:

1. two or three overlapping source photographs;
2. the one-command invocation;
3. generated depth maps;
4. a rotating colored PLY reconstruction;
5. a compact overlay: chip, runtime, precision, and “native MLX / no CUDA.”

Use only images whose publication rights are clear. Do not speed up footage in a
way that implies a false runtime.

**Acceptance gate:**

- the result is understandable with sound off in five seconds;
- text remains legible on mobile;
- the measured runtime agrees with a committed result;
- source-image attribution and consent are documented.

### Card R4.2 — Rewrite the README as the release front door

Required order:

1. one-sentence value proposition;
2. launch demo;
3. three-command quickstart;
4. supported outputs and example artifacts;
5. generated MLX-vs-MPS benchmark table;
6. generated numerical-parity table and methodology;
7. supported hardware/precision/view matrix;
8. limitations and omitted tracking head;
9. Python API and CLI reference;
10. checkpoint and source licensing;
11. upstream paper citation and “unofficial community port” notice.

**Acceptance gate:**

- a clean-room user reproduces one scene without undocumented help;
- every quantitative README claim maps to a committed result;
- all links and commands work at the release revision;
- README does not imply affiliation with Meta, VGGT authors, Apple, or MLX.

### Card R4.3 — Add lightweight CI and release checks

CI must not download the full checkpoint. Include:

- formatting/linting;
- unit tests;
- tiny deterministic PyTorch-to-MLX parity fixture where available;
- result-schema validation;
- generated-report freshness check;
- package build and contents inspection;
- CLI no-weight smoke test;
- secret and large-file checks.

Full-model parity and hardware benchmarks remain manually triggered on the
designated Apple-Silicon machine, with artifacts uploaded for review.

**Acceptance gate:** all clean-clone CI jobs pass and an intentionally stale
report or malformed result fails.

---

## R5 — Hugging Face and release publication

### Card R5.1 — Prepare the Hugging Face model page

Whether or not weights are published, the page should include:

- `library_name: mlx` and relevant pipeline/task tags;
- base model and exact revision;
- license and usage restrictions appropriate to the selected checkpoint;
- supported hardware and memory expectations;
- installation and one-scene example;
- output schema;
- benchmark and parity methodology with links to raw results;
- conversion procedure and checksums;
- limitations, tracking omission, and unofficial-port notice;
- citation for VGGT (`arxiv:2503.11651`) and this software release.

**Acceptance gate:** card renders correctly, download instructions are tested,
and artifact availability matches the distribution decision from R0.2.

### Card R5.2 — Publish release candidate `v0.1.0-rc1`

**Release contents:**

- signed or annotated tag;
- wheel/source distribution candidate;
- changelog;
- demo media;
- checksums;
- known limitations;
- reproducibility commands;
- final benchmark and parity result links.

**Acceptance gate:** a clean-room install reproduces one reconstruction and one
reported benchmark cell from the tagged revision.

### Card R5.3 — Publish `v0.1.0`

Publish GitHub release, PyPI package, and the approved Hugging Face artifacts in
a coordinated window. Request membership or mirroring in the Hugging Face
`mlx-community` organization only after the model page is complete.

**Acceptance gate:** public install links work, release artifacts match their
checksums, and no critical issue is found during the release-candidate review.

---

## R6 — Distribution campaign

### Card R6.1 — Ecosystem listings

Prepare narrowly scoped submissions for:

- MLX Community Projects discussion;
- `raullenchai/awesome-mlx`;
- `ruili3/awesome-dust3r`;
- `ziplab/Awesome-Feed-Forward-3D`;
- upstream VGGT discussion or issue offering the port as an unofficial community
  implementation.

Each submission should state exactly what is implemented, what is omitted, the
tested hardware, the license boundary, and the parity/benchmark evidence. Avoid
duplicate or promotional issues where a repository provides a designated
showcase or discussion channel.

### Card R6.2 — Technical launch article

Recommended structure:

1. phone photos to local 3D reconstruction;
2. why a native MLX port matters;
3. difficult implementation details: layouts, alternating attention, 2D RoPE,
   DPT resizing, and camera geometry;
4. how numerical parity was established;
5. honest MLX-versus-MPS benchmark methodology and result;
6. reduced-precision or Metal optimization story;
7. limitations, licensing, and reproducibility commands.

### Card R6.3 — Broadcast package

Prepare platform-specific posts for X, LinkedIn, Show HN, and
`r/MachineLearning`. Every post should lead with the demo and one defensible
claim, link directly to the repository, credit the VGGT authors, and avoid
calling the port official.

Suggested correctness-first copy:

> I ported VGGT-1B to native MLX for local 3D reconstruction on Apple Silicon.
> It predicts cameras, depth, confidence, and colored point clouds without
> PyTorch at inference—and the port is validated layer-by-layer against the
> reference model. Demo, benchmarks, and code: [link]

If and only if R2.4 establishes a speedup, replace the first sentence with the
precise bounded performance result.

### Card R6.4 — Post-launch response

For the first week:

- label installation, correctness, performance, and feature issues separately;
- reproduce bugs against the tagged release;
- publish corrections rather than silently changing benchmark claims;
- prioritize broken installation and corrupted output over new features;
- capture requested machines/view counts for the next benchmark matrix;
- invite independent reproduction of one benchmark cell.

---

## Launch blockers

Do not announce broadly while any of these is true:

- same-machine MPS comparison is missing or methodologically unequal;
- README contains a performance number without raw result evidence;
- weight redistribution status is unresolved but weights are publicly uploaded;
- fp32 parity is red;
- supported reduced precision produces non-finite or materially degraded output;
- clean installation requires undocumented manual repair;
- demo inputs lack clear publication rights;
- the release claims full VGGT while the tracking head remains omitted.

## Definition of done

The public-release project is complete when:

- fp32 parity remains green at the tagged revision;
- MLX-versus-MPS results are reproducible and schema-valid;
- every supported reduced-precision mode has a quality report;
- a clean user can install, acquire or convert weights, and reconstruct a scene
  using the CLI alone;
- package, model page, release artifacts, licenses, and checksums agree;
- the README is generated from or linked to committed evidence;
- launch media and posts make one accurate, defensible claim;
- upstream and MLX community submissions clearly identify the project as an
  unofficial community port.

## Recommended commit sequence

Use one commit per card:

```text
release: record fp32 baseline
docs: define checkpoint distribution policy
bench: add versioned result and parity schemas
bench: add equivalent PyTorch-MPS adapter
bench: add resumable comparison runner
bench: publish same-machine fp32 results
perf: add explicit reduced-precision modes
test: gate fp16 and bf16 output quality
perf: optimize measured MLX bottlenecks
cli: add setup and reconstruction commands
weights: add approved acquisition workflow
build: prepare PyPI distributions
docs: add launch demo and release README
ci: validate package results and reports
release: prepare v0.1.0 candidate
```

Stop after any failed correctness, license, or clean-install gate. Diagnose and
record the failure before continuing; publicity does not outrank trust.
