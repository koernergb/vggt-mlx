# VGGT → MLX Port — Cursor/Codex Task Cards

> **How to use this file.** Each card is one PR/commit. Give the agent **one card at a time**. A card is *done* only when its **Acceptance (gate)** is green — do **not** advance until it is. This ordering + hard gates is the whole point: it prevents a half-built scaffold. Cursor: run a card, review the diff, run the gate, commit. Codex: paste the card as the task; the acceptance command is the stop condition.

## Global rules (apply to every card — the agent should treat these as always-on)

1. **Float32 everywhere; validate parity on CPU.** Set `mx.set_default_device(mx.cpu)` in every parity test. MLX's fast matmul rounds fp32 operands to ~tf32 (10-bit mantissa) on GPU, so deep stacks drift — CPU removes that floor. Only try GPU / bf16 / fp16 *after* the fp32 gate is green, as a separate experiment. Never rely on autocast (VGGT breaks under MPS fp16 autocast).
2. **Weight layout conversion:**
   - `nn.Linear`: PyTorch stores `[out, in]`; MLX `Linear` uses the **same** `[out, in]` → **no transpose**. Shape-check to confirm.
   - `nn.Conv2d`: permute PyTorch `[out, in, kh, kw]` → MLX `[out, kh, kw, in]` with `(0,2,3,1)`.
   - `nn.ConvTranspose2d`: PyTorch stores `[in, out/groups, kh, kw]`; for the ungrouped layers used here, permute to MLX `[out, kh, kw, in]` with `(1,2,3,0)`.
   - Feature maps in MLX are **NHWC** (channels last). LayerNorm weight/bias map 1:1.
3. **Attention:** use `mx.fast.scaled_dot_product_attention(q, k, v, scale=..., mask=...)` (softmax runs in fp32). Use **additive** masks (large-negative at disallowed positions), shapes `(B, n_heads, T, D)`.
4. **Source of truth beats this file.** The `vggt.py`, `dpt_head.py`, and `camera_head.py` specs below are confirmed from upstream source. The **Aggregator literals** (`aa_order`, `aa_block_size`, `rope_freq`, `qk_norm`, exact `camera_token`/`register_token` shapes, `frame_blocks`/`global_blocks` container names) are *reconstructed* — **Card 0.2 dumps the real values and every M3 card must reconcile against that dump before coding.**
5. **One card = one commit.** Keep diffs scoped to the card's `Files`. If a card reveals the spec is wrong, fix the spec in this file in the *same* commit and note it.

## Dependency graph / build order

```
0.1 -> 0.2 -> 0.3 -> 1.1 -> 1.2 -> 1.3 -> 2.1 -> 2.2
                                                    |
        +-------------------------------------------+
        v
3.1 -> 3.2 -> 3.3 -> 3.4 -> 3.5 -> 4.1 -+
                                        +-> 4.4 -> 5.1 -> 5.2 -> 5.3 -> 5.4
                                 4.2 -> 4.3 -+
```

Reference model config (confirmed): `img_size=518, patch_size=14, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, num_register_tokens=4, patch_start_idx=5`. Backbone = DINOv2 **ViT-L/14 with registers**. Aggregator concatenates frame+global outputs → heads see `dim_in=2048`.

---

## MILESTONE 0 — Reference & Harness

### Card 0.1 — Repo bootstrap & skeleton
**Depends on:** none · **Milestone:** M0

**Objective:** Create the full package tree with importable (stub) modules and tooling config, so every later card has a home.

**Files (create):**
```
vggt-mlx/
├── pyproject.toml            # deps: mlx>=0.31, numpy, pillow, huggingface_hub, pytest; dev: torch (Colab only)
├── .gitignore                # weights/, *.npz, __pycache__, .venv
├── README.md                 # placeholder; filled in Card 5.4
├── vggt_mlx/__init__.py
├── vggt_mlx/config.py                    # stub (Card 1.1 fills)
├── vggt_mlx/layers/{__init__,attention,rope2d,block,patch_embed,mlp}.py
├── vggt_mlx/models/{__init__,aggregator,vggt}.py
├── vggt_mlx/heads/{__init__,camera_head,dpt_head}.py
├── vggt_mlx/utils/{__init__,pose_enc,geometry,load_fn}.py
├── vggt_mlx/convert/{__init__,torch_to_mlx}.py
├── tools/oracle.py           # stub (Card 0.3 fills)
├── tests/__init__.py
├── tests/fixtures/.gitkeep
└── demo.py                   # stub prints "not implemented" (Card 5.3 fills)
```

**Steps:**
1. Each stub module defines the class/function *signatures* named in later cards with `raise NotImplementedError` bodies, so imports succeed.
2. `pyproject.toml` pins `mlx>=0.31`; put `torch` under a `dev`/`colab` extra (it won't install on the Mac path).
3. Add a `Makefile` or `pyproject` script alias `test = "pytest -q"`.

**Pitfalls:** don't write real logic here — signatures only. Don't add `torch` to the base deps.

**Acceptance (gate):**
- [ ] `python -c "import vggt_mlx, vggt_mlx.models.vggt, vggt_mlx.heads.dpt_head, vggt_mlx.layers.attention"` exits 0.
- [ ] `pytest -q` runs (0 tests, no import errors).

---

### Card 0.2 — Lock ground truth from upstream source (Colab/T4)
**Depends on:** 0.1 · **Milestone:** M0 · **Runs on Colab (needs torch + GPU download)**

**Objective:** Replace *all* reconstructed literals with verified ones read from the real model, and record every state-dict key + submodule repr for the converter and parity tests.

**Files (create):** `tools/introspect.py`, `tests/fixtures/state_dict_keys.txt`, `tests/fixtures/module_repr.txt`, `tests/fixtures/ARCH_NOTES.md`

**Steps:**
1. `pip install vggt` (or clone `facebookresearch/vggt`), then run:
   ```python
   python tools/introspect.py
   ```
   Use `python tools/introspect.py --no-pretrained` for an architecture-only
   smoke test that does not download the checkpoint.
2. From the repr + `aggregator.py` source, **write down the real values** into `ARCH_NOTES.md`: `aa_order`, `aa_block_size`, `rope_freq`, whether `qk_norm` is used, LayerScale `init_values`, the exact **shapes** of `camera_token` and `register_token`, and the **container names** for the frame vs global block lists (e.g. `frame_blocks` / `global_blocks` vs something else).
3. Note the two-slot special-token trick (index 0 = reference frame, index 1 = other frames) and confirm `patch_start_idx`.

**Pitfalls:** the model download is ~5 GB (gated? use the non-commercial `facebook/VGGT-1B`). Do this on Colab, commit only the small text files (not weights). The default patch embedder is the full DINOv2 ViT-L/14-with-registers module (including its 24 blocks), not only its projection convolution.

**Acceptance (gate):**
- [ ] `state_dict_keys.txt` and `module_repr.txt` committed and non-empty.
- [ ] `ARCH_NOTES.md` lists concrete values for every literal in Global Rule #4 (no "TBD" left).

---

### Card 0.3 — Oracle fixture generator (Colab/T4)
**Depends on:** 0.2 · **Milestone:** M0 · **Runs on Colab**

**Objective:** Dump PyTorch intermediate activations in **fp32 (autocast disabled)** for 1-image and 3-image inputs, to serve as parity oracles for all later tests.

**Files (edit):** `tools/oracle.py` · **(create):** `tests/fixtures/oracle_1view.npz`, `tests/fixtures/oracle_3view.npz`, `tests/fixtures/sample_images/` (2–3 small demo jpgs)

**Steps:**
1. Load `facebook/VGGT-1B`, run **without** `torch.cuda.amp.autocast` (force fp32), on CPU if possible for a clean reference.
2. Register forward hooks and save: preprocessed input tensor; post-`patch_embed` tokens; aggregator outputs after layers **4, 11, 17, 23**; camera head `pose_enc`; final `depth`, `depth_conf`; `extrinsic`, `intrinsic`. Save arrays as numpy in a single `.npz` per view-count. Also save the exact preprocessed pixel tensor so the MLX side feeds identical input.
3. Keep images tiny (downscale) so fixtures stay < ~50 MB total.

**Pitfalls:** must match the MLX preprocessing exactly later — save the *preprocessed* tensor, not the raw image. Disable autocast or parity thresholds won't hold.

**Acceptance (gate):**
- [ ] Both `.npz` files load and contain keys: `input, patch_embed, agg_l4, agg_l11, agg_l17, agg_l23, pose_enc, depth, depth_conf, extrinsic, intrinsic`.
- [ ] Shapes recorded in `ARCH_NOTES.md`.

---

## MILESTONE 1 — Config & Weight Conversion

### Card 1.1 — Config dataclass
**Depends on:** 0.2 · **Milestone:** M1

**Objective:** Single source of hyperparameters, populated from verified values.

**Files (edit):** `vggt_mlx/config.py`

**Spec:** `@dataclass VGGTConfig` with `img_size=518, patch_size=14, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0, num_register_tokens=4, patch_start_idx=5, rope_freq=<from 0.2>, aa_order=<from 0.2>, aa_block_size=<from 0.2>, layerscale_init=<from 0.2>, intermediate_layer_idx=(4,11,17,23), dpt_features=256, dpt_out_channels=(256,512,1024,1024), pose_encoding_type="absT_quaR_FoV", pose_dim=9`.

**Acceptance (gate):**
- [ ] `pytest tests/test_config.py` — asserts field values match `ARCH_NOTES.md` (write this 5-line test).

---

### Card 1.2 — PyTorch→MLX weight converter
**Depends on:** 1.1 · **Milestone:** M1

**Objective:** Turn `facebook/VGGT-1B` weights into an MLX-native safetensors with remapped keys and permuted conv weights.

**Files (edit):** `vggt_mlx/convert/torch_to_mlx.py`

**Spec / steps:**
1. Load the torch state dict (from the `.pt`/`safetensors` downloaded on Colab, or via `huggingface_hub`).
2. Build `{torch_key: mlx_key}`. Prefixes stay the same (`aggregator.`, `camera_head.`, `depth_head.`, `point_head.`; **drop** `track_head.*` — out of scope). Only rename where MLX module attribute names differ from your implementation (reconcile against `state_dict_keys.txt`).
3. Per-key transform: **Linear → identity** (verify `[out,in]`); **Conv2d → permute** `(0,2,3,1)`; **ConvTranspose2d → permute** `(1,2,3,0)`; LayerNorm/params → identity.
4. Write `weights/vggt-1b-mlx.safetensors` via `mx.save_safetensors`.
5. Emit a `conversion_report.txt`: count of keys mapped, skipped (track head), and any shape mismatches.

**Pitfalls:** a Linear you accidentally transpose will pass shape checks but fail parity — assert exact shape equality against `state_dict_keys.txt` rather than transposing "to be safe". Distinguish Conv2d (has 4-D weight) from Linear (2-D) programmatically.

**Acceptance (gate):**
- [ ] Script produces `weights/vggt-1b-mlx.safetensors`.
- [ ] `conversion_report.txt` shows 0 unmapped non-track keys and 0 shape mismatches.

---

### Card 1.3 — Load-weights smoke test
**Depends on:** 1.2, plus stubs from 2.1/3.*/4.* as they land (run last in M1 or revisit)

**Objective:** Prove the converted file loads into the assembled MLX model with `strict=True` once modules exist. (If modules aren't built yet, gate on a partial load of the backbone only, then re-run after M4.)

**Files (create):** `tests/test_weight_load.py`

**Acceptance (gate):**
- [ ] `model.load_weights(list(mx.load("weights/vggt-1b-mlx.safetensors").items()), strict=True)` raises **no** missing/unexpected-key error (backbone-only acceptable until M4; full model required before M5).

---

## MILESTONE 2 — Patch Embedder (DINOv2 backbone)

### Card 2.1 — PatchEmbed + DINOv2 ViT-L/14 blocks
**Depends on:** 1.1 · **Milestone:** M2

**Objective:** Reimplement the DINOv2 patch embedder used as VGGT's tokenizer. **Reuse `mlx-image`'s DINOv2 ViT-L/14 modules** as the starting point rather than writing from scratch.

**Files (edit):** `vggt_mlx/layers/patch_embed.py`, `vggt_mlx/layers/mlp.py`

**Spec:** `proj = nn.Conv2d(3, 1024, kernel_size=14, stride=14)` (NHWC), norm = Identity, 24 internal transformer blocks (embed 1024, 16 heads, mlp_ratio 4, LayerScale). Input 518×518 → 37×37 = **1369** patch tokens. Confirm block internals against `mlx-image` and `module_repr.txt`.

**Pitfalls:** MLX Conv2d expects NHWC input and `[out,kh,kw,in]` weight (already handled by Card 1.2). Don't add DINOv2's CLS/register handling here — VGGT injects its own special tokens in the Aggregator (Card 3.4).

**Acceptance (gate):** deferred to 2.2.

---

### Card 2.2 — Patch-embed parity test
**Depends on:** 2.1, 0.3, 1.2 · **Milestone:** M2

**Objective:** First real parity checkpoint.

**Files (create):** `tests/test_patch_embed_parity.py`

**Steps:** load `oracle_1view.npz` `input`, run MLX patch embedder (CPU, fp32), compare to `patch_embed` oracle via `max(abs(a-b))`.

**Acceptance (gate):**
- [ ] `max_abs_diff < 1e-4` on `oracle_1view` and `oracle_3view`.

---

## MILESTONE 3 — Aggregator (alternating attention)

> **Reconcile every card in this milestone against `ARCH_NOTES.md` / `state_dict_keys.txt` first.** These are the reconstructed literals.

### Card 3.1 — 2D Rotary Position Embedding + PositionGetter
**Depends on:** 1.1, 0.2 · **Milestone:** M3

**Objective:** Implement VGGT's **axial 2D RoPE** (separate x/y rotation over half the head dims each) and the `(row, col)` index generator.

**Files (edit):** `vggt_mlx/layers/rope2d.py`

**Spec:** `RotaryPositionEmbedding2D(frequency=<rope_freq from 0.2>)` applies rotation to Q/K and infers the head dimension at call time; `PositionGetter` returns per-patch `(row, col)` on the 37×37 grid. Split head_dim in half → row/y frequencies followed by column/x frequencies, matching upstream `rope.py`.

**Pitfalls:** wrong frequency base or swapped x/y axis **silently** degrades geometry (no crash). Add a standalone unit test: rotating by known angles matches a hand-computed reference for a 4-dim toy case.

**Acceptance (gate):**
- [ ] `pytest tests/test_rope2d.py` — toy-case rotation matches analytic values to 1e-6.

---

### Card 3.2 — Fused-QKV attention (frame + global)
**Depends on:** 3.1 · **Milestone:** M3

**Objective:** One attention module matching the checkpoint's `attn.qkv.weight` / `attn.proj.weight` layout. As in upstream, the aggregator selects frame-wise versus global scope by reshaping the input stream before calling the shared module.

**Files (edit):** `vggt_mlx/layers/attention.py`

**Spec / steps:**
1. `qkv = Linear(1024, 3072, bias=True)`; split → q,k,v; reshape to `(B, n_heads=16, T, 64)`.
2. Apply 2D RoPE to q,k (Card 3.1).
3. Call `mx.fast.scaled_dot_product_attention(q,k,v, scale=1/sqrt(64), mask=mask)`; `proj = Linear(1024,1024)`.
4. **Frame scope:** the aggregator supplies `(B*S,P,C)`, batching frames independently. **Global scope:** it supplies `(B,S*P,C)`, allowing all frame tokens to attend jointly. No mask is needed for either upstream path.
5. Respect `qk_norm` if `ARCH_NOTES.md` says it's on.

**Pitfalls:** if an explicit mask is added for a future path, use an additive `-inf`/large-negative mask rather than a boolean mask. The released upstream frame/global paths use reshaping and no mask. Keep everything fp32.

**Acceptance (gate):**
- [ ] `pytest tests/test_attention.py` — frame-mode on a 2-frame toy input equals a manual per-frame reference to 1e-5; global-mode equals full-attention reference.

---

### Card 3.3 — Transformer Block (LayerScale + MLP)
**Depends on:** 3.2 · **Milestone:** M3

**Objective:** The repeating block: `x = x + ls1*attn(norm1(x)); x = x + ls2*mlp(norm2(x))`.

**Files (edit):** `vggt_mlx/layers/block.py`

**Spec:** pre-norm LayerNorm; `Mlp(1024→4096→1024)` GELU; LayerScale `init_values=<from 0.2>` (per-channel learned scale). The block accepts positions and operates on the stream shape prepared by the aggregator; it does not receive a frame/global mode flag.

**Acceptance (gate):** deferred to 3.5 (block is exercised inside the Aggregator parity test).

---

### Card 3.4 — Aggregator assembly
**Depends on:** 3.3, 2.1 · **Milestone:** M3

**Objective:** Wire patch embedder + special tokens + 24×(frame,global) alternation into the Aggregator, returning the per-layer token list.

**Files (edit):** `vggt_mlx/models/aggregator.py`

**Spec / steps:**
1. Learned params `camera_token` and `register_token` with the **two-slot** shapes from 0.2 (slot 0 = reference/first frame, slot 1 = broadcast to all others). Prepend **5** special tokens per frame (`patch_start_idx=5`).
2. Loop `depth=24`; per layer apply order `aa_order` with `aa_block_size` (from 0.2 — typically one frame block then one global block).
3. Collect the output tokens **after each layer** into `aggregated_tokens_list` (len 24). Each entry is fed later to the DPT heads at indices 4/11/17/23. Return `(aggregated_tokens_list, patch_start_idx)`.
4. Single-image case: global attention degenerates to frame attention — must still run.

**Pitfalls:** the concat that makes heads see 2048 dims happens here (frame-attn output ‖ global-attn output). Get the concat axis/order right or every head misaligns. Unified-memory: 4 views ≈ 5.5k tokens is fine on 16 GB; don't test with 20+ views.

**Acceptance (gate):** deferred to 3.5.

---

### Card 3.5 — Aggregator parity test
**Depends on:** 3.4, 0.3 · **Milestone:** M3 · **This is the make-or-break gate.**

**Files (create):** `tests/test_aggregator_parity.py`

**Steps:** feed oracle `input`, compare MLX aggregator outputs at layers 4/11/17/23 to `agg_l4/l11/l17/l23` (CPU, fp32), for both 1-view and 3-view.

**Pitfalls / triage:** if this fails > 1e-2, in order of likelihood: (a) RoPE frequency/axis wrong; (b) fused-qkv split order wrong; (c) frame-vs-global concat axis wrong; (d) special-token slot assignment (ref vs others) swapped. Fix before touching heads.

**Acceptance (gate):**
- [ ] `max_abs_diff < 1e-3` at all four layers for `oracle_1view` **and** `oracle_3view`.

---

## MILESTONE 4 — Prediction Heads

### Card 4.1 — DPT head (depth + point), source-confirmed
**Depends on:** 3.5 · **Milestone:** M4

**Objective:** Implement the shared DPT decoder; instantiate twice (depth `output_dim=2`, point `output_dim=4`).

**Files (edit):** `vggt_mlx/heads/dpt_head.py`

**Spec (confirmed from upstream `dpt_head.py`):**
- `norm = LayerNorm(2048)`; slice `tokens[:, :, patch_start_idx:]` (drop 5 special tokens).
- `projects = [Conv2d(2048→oc, k=1) for oc in (256,512,1024,1024)]`.
- `resize_layers = [ConvTranspose2d(256,256,k=4,s=4), ConvTranspose2d(512,512,k=2,s=2), Identity(), Conv2d(1024,1024,k=3,s=2,p=1)]` — reassemble layers 4/11/17/23 into a 4-level pyramid.
- `scratch.refinenet1..4` fusion (refinenet4 `has_residual=False`); `output_conv1 = Conv2d(256→128,k=3,p=1)`; `output_conv2 = Sequential(Conv2d(128→32,k=3,p=1), ReLU, Conv2d(32→output_dim,k=1))`.
- **depth:** `output_dim=2` (depth+conf), `activation="exp"`, `conf_activation="expp1"` → `depth [B,S,H,W,1]`, `depth_conf [B,S,H,W]`.
- **point:** `output_dim=4` (xyz+conf), `activation="inv_log"` → `world_points [B,S,H,W,3]`, `world_points_conf`.

**Pitfalls:** all convs NHWC + permuted weights (Card 1.2). MLX bilinear `Upsample` vs PyTorch `F.interpolate(align_corners=False)` can differ sub-pixel — test the upsample step in isolation and match modes. Get the `exp` vs `inv_log` activation on the right head.

**Acceptance (gate):** deferred to 4.4.

---

### Card 4.2 — Camera head (AdaLN, iterative refine), source-confirmed
**Depends on:** 3.5 · **Milestone:** M4

**Objective:** Regress the 9-D pose encoding from the camera token.

**Files (edit):** `vggt_mlx/heads/camera_head.py`

**Spec (confirmed from upstream `camera_head.py`):** `trunk_depth=4`, `pose_encoding_type="absT_quaR_FoV"`, pose dim **9** = T(3) + quat(4, XYZW scalar-last) + FoV(2). `embed_pose = Linear(9→2048)`; `poseLN_modulation = Sequential(SiLU, Linear(2048→6144))`; `adaln_norm = LayerNorm(2048, eps=1e-6, affine=False)`; 4-block trunk applied over **4 refinement iterations**; `pose_branch = Mlp(2048→1024→9)`. Input is the camera token stream from the aggregator.

**Pitfalls:** quaternion order is scalar-last (XYZW) — don't assume WXYZ. AdaLN produces 6 modulation params (shift/scale/gate ×2) from the 6144 projection — split correctly.

**Acceptance (gate):** deferred to 4.4.

---

### Card 4.3 — Pose→camera + depth unprojection utils
**Depends on:** 4.2 · **Milestone:** M4

**Objective:** Convert pose encoding to OpenCV camera matrices and unproject depth to a point cloud (the demo's 3D output — more accurate than the point head per the paper).

**Files (edit):** `vggt_mlx/utils/pose_enc.py`, `vggt_mlx/utils/geometry.py`

**Spec:** `pose_encoding_to_extri_intri(pose_enc, image_hw) → extrinsic [B,S,3,4], intrinsic [B,S,3,3]`; `unproject_depth_map_to_point_map(depth, extrinsic, intrinsic) → points [B,S,H,W,3]`. Port the math directly from upstream `utils/`.

**Acceptance (gate):**
- [ ] `pytest tests/test_pose_enc.py` — `extrinsic`/`intrinsic` from the oracle `pose_enc` match oracle `extrinsic`/`intrinsic` to 1e-4.

---

### Card 4.4 — Heads parity test
**Depends on:** 4.1, 4.2, 4.3, 0.3 · **Milestone:** M4

**Files (create):** `tests/test_heads_parity.py`

**Steps:** run aggregator → depth head + camera head on oracle input (CPU, fp32); compare `pose_enc`, `depth`, `depth_conf` to oracle.

**Acceptance (gate):**
- [ ] `pose_enc` `max_abs_diff < 1e-3`; `depth` `max_abs_diff < 1e-2`; `depth_conf` finite and correlated.

---

## MILESTONE 5 — Assembly & Demo

### Card 5.1 — Image preprocessing
**Depends on:** 1.1 · **Milestone:** M5

**Objective:** Match upstream `load_and_preprocess_images` exactly (so inputs equal the oracle).

**Files (edit):** `vggt_mlx/utils/load_fn.py`

**Spec:** isotropic resize long side → 518, crop to a multiple of 14, normalize with DINOv2 mean/std, output NHWC MLX array `[S,H,W,3]`. Verify against the saved oracle `input` tensor.

**Acceptance (gate):**
- [ ] `pytest tests/test_preprocess.py` — preprocessing the raw sample images reproduces the oracle `input` tensor to 1e-5.

---

### Card 5.2 — VGGT top-level model + forward
**Depends on:** 3.4, 4.1, 4.2, 4.3 · **Milestone:** M5

**Objective:** Assemble `VGGT` (aggregator + camera_head + depth_head + point_head; **no track_head**) with a `forward(images)` returning `{pose_enc, depth, depth_conf, world_points, world_points_conf, extrinsic, intrinsic}`.

**Files (edit):** `vggt_mlx/models/vggt.py`

**Then re-run Card 1.3's gate with the full model and `strict=True`.**

**Acceptance (gate):**
- [ ] Full-model `load_weights(..., strict=True)` passes (no missing/unexpected keys).
- [ ] `pytest tests/test_end_to_end_parity.py` — full forward on oracle input matches `depth` to < 1e-2.

---

### Card 5.3 — One-command demo
**Depends on:** 5.1, 5.2 · **Milestone:** M5 · **The "lovable" payoff.**

**Objective:** `python demo.py img1.jpg img2.jpg` → writes a depth-map PNG per image + a fused point-cloud `.ply`, and prints per-frame timing.

**Files (edit):** `demo.py`

**Steps:** auto-download weights from HF if absent → preprocess → forward → colorize depth (matplotlib/Pillow) → unproject depth to points (Card 4.3) → write `output/depth_*.png` and `output/points.ply` → print `ms/frame`.

**Pitfalls:** cap the demo at ≤4 views (16 GB unified memory). Runs in fp32.

**Acceptance (gate):**
- [ ] On a MacBook, `python demo.py <two sample imgs>` produces a viewable depth PNG and a `.ply` that opens in a mesh viewer, and prints a timing number.

---

### Card 5.4 — README, benchmark, license
**Depends on:** 5.3 · **Milestone:** M5

**Objective:** Ship-ready repo front door — this is what earns stars and reads as portfolio evidence.

**Files (edit):** `README.md`

**Must include:**
1. One-line pitch ("Run the CVPR 2025 best-paper 3D model natively on Apple Silicon — no CUDA, no PyTorch at inference"), demo GIF, one-command quickstart.
2. **Benchmark table**: ms/frame on your M-series (fp32) vs PyTorch-MPS baseline, plus the parity numbers (max-abs-diff per stage) — the credibility signal.
3. **Parity methodology** paragraph (oracle fixtures, CPU-validated).
4. **License note:** original `facebook/VGGT-1B` weights are **CC-BY-NC-4.0 (non-commercial)** — ship the **conversion script + instructions, not converted weights**, or use the gated `VGGT-1B-Commercial` checkpoint and honor its "no military use" terms. State this clearly.
5. Scope note: track head intentionally omitted (link it as future work).

**Acceptance (gate):**
- [ ] README renders with quickstart, benchmark+parity table, and the license section; a fresh clone can reproduce the demo by following it.

---

## Stretch (do not start until 5.4 ships)
- **S1** point-head polish and parity. **S2** bf16/fp16 + GPU perf pass (separate from fp32 gates). **S3** track head. **S4** publish MLX weights to `mlx-community` (license permitting) + launch post following the RF-DETR playbook; link from any upstream "does this run on Apple Silicon?" issue.

> **Reminder:** the moment a card's gate is green, commit and move on. If a gate won't go green after ~3 focused attempts, that's a signal to re-read the upstream source for that specific module (not to grind) — the oracle diff tells you which layer is wrong.
