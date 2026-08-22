<div align="center">

# vggt-mlx

### VGGT-1B, running natively on Apple Silicon

**Reconstruct camera poses, depth, confidence, and dense 3D geometry with MLX—no CUDA or PyTorch at inference time.**

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![MLX 0.31+](https://img.shields.io/badge/MLX-0.31%2B-111111?logo=apple&logoColor=white)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-native-000000?logo=apple&logoColor=white)
![Tests](https://img.shields.io/badge/tests-43%20passing-2ea44f)

[Quickstart](#quickstart) · [Python API](#python-api) · [Results](#performance-and-parity) · [Model terms](#model-weights-and-terms)

</div>

<p align="center">
  <img src="docs/demo.gif" alt="VGGT-MLX depth prediction" width="820" />
</p>

`vggt-mlx` is an fp32 [MLX](https://github.com/ml-explore/mlx) port of Facebook Research's [VGGT](https://github.com/facebookresearch/vggt), the CVPR 2025 best-paper model for feed-forward 3D reconstruction. Give it one to four related images and it returns camera parameters, depth, confidence, and world-space points.

> [!IMPORTANT]
> This repository ships conversion code, not model weights. The original `facebook/VGGT-1B` checkpoint is non-commercial; read [Model weights and terms](#model-weights-and-terms) before use.

## What you get

| Capability | Output |
|---|---|
| Camera regression | 9-D pose encoding, OpenCV extrinsics, and intrinsics |
| Dense depth | Per-pixel depth and confidence |
| 3D reconstruction | Dense world-space points and confidence |
| One-command demo | Colorized depth PNGs and a fused PLY point cloud |
| Verified conversion | Strict loading of all 1,403 in-scope checkpoint tensors |

The aggregator, DINOv2 backbone, camera head, DPT depth head, point head, pose conversion, preprocessing, and demo are implemented in MLX. The upstream tracking head is intentionally out of scope for this release.

## Quickstart

### 1. Install

Requirements: Apple Silicon Mac, macOS, Python 3.10+, and enough disk space for the original and converted checkpoints.

```bash
git clone https://github.com/koernergb/vggt-mlx.git
cd vggt-mlx

python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

### 2. Install upstream VGGT and convert the weights

```bash
UPSTREAM_VGGT_DIR="$(mktemp -d /tmp/facebook-vggt.XXXXXX)"

git clone --depth 1 \
  https://github.com/facebookresearch/vggt.git \
  "$UPSTREAM_VGGT_DIR"

python -m pip install -e "$UPSTREAM_VGGT_DIR"

python -c "from vggt.models.vggt import VGGT; print('Upstream VGGT installed successfully')"

python -m vggt_mlx.convert.torch_to_mlx
```

Using a freshly generated temporary directory makes this setup safe to rerun, even if an older or incomplete VGGT checkout already exists under `/tmp`. The verification command must print `Upstream VGGT installed successfully` before conversion begins.

The converter downloads the official checkpoint from Hugging Face, reads it lazily, converts PyTorch convolution layouts, omits the tracking head, and creates:

```text
weights/
├── conversion_report.txt
└── vggt-1b-mlx.safetensors
```

A valid conversion reports **0 unmapped non-tracking keys** and **0 shape mismatches**. Converted weights remain local and are ignored by Git.

> [!NOTE]
> The Hugging Face unauthenticated-request message is only a rate-limit warning. Set `HF_TOKEN` if desired; it is not required for the public non-commercial checkpoint.

### 3. Reconstruct a scene

Use photos of the **same scene from different viewpoints**:

```bash
python demo.py image-1.jpg image-2.jpg
```

```text
output/
├── depth_00.png
├── depth_01.png
└── points.ply
```

The PLY contains a fused, colored point cloud and opens in tools such as MeshLab or CloudCompare. The demo caps input at four views to stay practical on a 16 GB Mac.

## Python API

```python
import mlx.core as mx

from vggt_mlx import VGGT
from vggt_mlx.utils.load_fn import load_and_preprocess_images

images = load_and_preprocess_images([
    "image-1.jpg",
    "image-2.jpg",
])

model = VGGT()
model.load_weights(
    list(mx.load("weights/vggt-1b-mlx.safetensors").items()),
    strict=True,
)
model.eval()

prediction = model(images)
mx.eval(*prediction.values())
```

`prediction` contains:

| Key | Meaning |
|---|---|
| `pose_enc` | Translation, XYZW quaternion, and field of view |
| `extrinsic` | OpenCV world-to-camera matrix `[B,S,3,4]` |
| `intrinsic` | Camera intrinsic matrix `[B,S,3,3]` |
| `depth` / `depth_conf` | Dense depth and confidence |
| `world_points` / `world_points_conf` | Dense 3D points and confidence |

## Performance and parity

### Apple Silicon benchmark

Measured on an **Apple M4 with 16 GB unified memory**, MLX 0.32, fp32 weights, and two 126×518 views. Timing covers model forward execution after loading and preprocessing.

| Runtime | Precision | Views | Time |
|:---|:---:|---:|---:|
| **MLX 0.32** | fp32 | 2 | **1,949.3 ms/frame** |
| PyTorch MPS | fp32 | 2 | Not yet measured |

The PyTorch-MPS result is intentionally left unclaimed until it is reproduced on the same machine and inputs.

### Numerical parity

| Stage | Worst observed result | Acceptance gate |
|:---|---:|---:|
| DINO patch tokens | **1.38e-4** max abs diff | 2e-4 |
| Aggregator layer 23 | **1.41e-3** max abs diff | 2e-3 |
| Camera pose | **3.88e-7** max abs diff | 1e-3 |
| Depth head | **1.04e-5** max abs diff | 1e-2 |
| Depth confidence | **>0.99999999999** correlation | >0.99 |

Oracle fixtures were generated from the official PyTorch checkpoint in fp32 with autocast disabled. The suite validates one-view and three-view inputs, captured intermediate activations, architecture literals, checkpoint conversion, strict loading, camera geometry, preprocessing, prediction heads, and complete inference. Gates account for the small expected drift between the CUDA T4 oracle, PyTorch CPU, and MLX kernels.

Run the full suite:

```bash
pytest -q
```

Current result: **43 passed**.

## Architecture

```text
images
  └─ DINOv2 ViT-L/14 patch backbone
      └─ 24 alternating frame/global attention layers
          ├─ camera head ── pose, extrinsics, intrinsics
          ├─ depth DPT ──── depth + confidence
          └─ point DPT ──── world points + confidence
```

Implementation milestones, acceptance gates, and upstream architecture notes live in [`docs/TASKS.md`](docs/TASKS.md) and [`tests/fixtures/ARCH_NOTES.md`](tests/fixtures/ARCH_NOTES.md).

## Model weights and terms

The source in this repository is distributed under the [VGGT License](LICENSE.txt), including its
acceptable-use restrictions and redistribution requirements.

Model weights are **not distributed** by this repository and retain their original terms:

- `facebook/VGGT-1B` is licensed under **CC-BY-NC-4.0** and is restricted to non-commercial use.
- Commercial users must obtain the gated `VGGT-1B-Commercial` checkpoint and comply with its additional restrictions, including its prohibition on military use.

Always review the current upstream model card and license before downloading, converting, or deploying a checkpoint.

## Scope and roadmap

The implementation plan for benchmarking, reduced precision, packaging,
Hugging Face distribution, and the public launch is in
[`docs/RELEASE_ROADMAP.md`](docs/RELEASE_ROADMAP.md).

- [x] DINOv2 patch backbone
- [x] Alternating frame/global aggregator
- [x] Camera, depth, and point heads
- [x] Pose conversion and depth unprojection
- [x] Weight conversion and strict-loading gates
- [x] One-command depth and PLY demo
- [ ] Upstream tracking head
- [ ] fp16/bf16 performance pass
- [ ] Reproducible PyTorch-MPS comparison

---

<div align="center">
Built for native 3D inference on Apple Silicon.
</div>
