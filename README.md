# vggt-mlx

Run the CVPR 2025 best-paper 3D model natively on Apple Silicon — no CUDA and no PyTorch at inference time.

![VGGT-MLX depth output](docs/demo.gif)

`vggt-mlx` is an fp32 MLX port of Facebook Research's VGGT-1B. It predicts camera poses, depth, confidence, and dense world points from one to four related images on an Apple Silicon Mac.

## Quickstart

Python 3.10+, macOS, and Apple Silicon are required. Converted model weights are intentionally not distributed by this repository; the setup below downloads the official checkpoint and converts it locally.

```bash
git clone https://github.com/koernergb/vggt-mlx.git
cd vggt-mlx
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

git clone --depth 1 https://github.com/facebookresearch/vggt.git /tmp/upstream-vggt
pip install -e /tmp/upstream-vggt
python -m vggt_mlx.convert.torch_to_mlx

python demo.py image-1.jpg image-2.jpg
```

The demo writes `output/depth_00.png`, one depth visualization per input, and a fused `output/points.ply`. Inputs must show the same scene from different viewpoints. Use at most four views on a 16 GB machine.

## Python API

```python
import mlx.core as mx
from vggt_mlx import VGGT
from vggt_mlx.utils.load_fn import load_and_preprocess_images

images = load_and_preprocess_images(["image-1.jpg", "image-2.jpg"])
model = VGGT()
model.load_weights(list(mx.load("weights/vggt-1b-mlx.safetensors").items()), strict=True)
model.eval()

prediction = model(images)
mx.eval(*prediction.values())
```

The output dictionary contains `pose_enc`, `depth`, `depth_conf`, `world_points`, `world_points_conf`, `extrinsic`, and `intrinsic`.

## Benchmark and parity

Measured on an Apple M4 with 16 GB unified memory, using two 126×518 views and fp32 weights. Timing covers model forward execution after loading and preprocessing.

| Runtime | Precision | Views | Time |
|---|---:|---:|---:|
| MLX 0.32 | fp32 | 2 | 1,949.3 ms/frame |
| PyTorch MPS | fp32 | 2 | Not measured |

The PyTorch-MPS cell is deliberately left unclaimed until a reproducible baseline is run on the same machine and inputs.

| Parity stage | Worst observed result | Gate |
|---|---:|---:|
| DINO patch tokens | 1.38e-4 max abs diff | 2e-4 |
| Aggregator layer 23 | 1.41e-3 max abs diff | 2e-3 |
| Camera pose | 3.88e-7 max abs diff | 1e-3 |
| Depth head | 1.04e-5 max abs diff | 1e-2 |
| Depth confidence | >0.99999999999 correlation | >0.99 |

Parity fixtures were produced from the official PyTorch `facebook/VGGT-1B` checkpoint with autocast disabled. Tests use saved one-view and three-view fp32 inputs and intermediate activations. Architecture literals, checkpoint keys, strict loading, camera matrices, preprocessing, prediction heads, and the full forward are covered by `pytest`. Cross-device gates allow the small expected drift between the CUDA T4 oracle, PyTorch CPU, and MLX kernels.

## Weight conversion

The converter reads the official safetensors checkpoint lazily, permutes convolution layouts, omits the tracking head, and writes:

- `weights/vggt-1b-mlx.safetensors`
- `weights/conversion_report.txt`

A successful report contains zero unmapped non-tracking keys and zero shape mismatches. The converted weights directory is gitignored.

## License and model terms

The code in this repository is provided under the repository license. Model weights retain their original terms and are not included here.

The `facebook/VGGT-1B` weights are licensed **CC-BY-NC-4.0** and are restricted to non-commercial use. Users needing commercial terms must obtain the gated `VGGT-1B-Commercial` checkpoint and comply with its additional restrictions, including its prohibition on military use. Review the current model card and license before downloading, converting, or deploying either checkpoint.

## Scope

This release ports the aggregator, camera head, depth head, point head, pose conversion, preprocessing, and demo. The upstream tracking head is intentionally omitted to keep the first release focused and practical on 16 GB Macs; it remains future work.

Development is organized as independently gated cards in [`docs/TASKS.md`](docs/TASKS.md). Run all validation with:

```bash
pytest -q
```
