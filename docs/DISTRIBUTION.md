# Distribution and Attribution Policy

**Status:** approved for the `v0.1` code release; model-weight redistribution is
not approved.

This document records the project's conservative publication decisions as of
August 21, 2026. It is an engineering release policy, not legal advice. Review
the current upstream terms again immediately before every public release.

## Authoritative upstream sources

- VGGT source and license: <https://github.com/facebookresearch/vggt>
- Original checkpoint: <https://huggingface.co/facebook/VGGT-1B>
- Commercial checkpoint: <https://huggingface.co/facebook/VGGT-1B-Commercial>
- Paper: <https://arxiv.org/abs/2503.11651>
- Pinned original-checkpoint revision used for the fp32 baseline:
  `860abec7937da0a4c03c41d3c269c366e82abdf9`

The repository includes an unmodified copy of the VGGT License v1, last updated
July 29, 2025, in [`LICENSE.txt`](../LICENSE.txt). That agreement covers Meta's
research materials and derivative works, requires redistributed derivatives to
remain under its terms with a copy of the agreement, requires research
publication acknowledgement, and incorporates its Acceptable Use Policy.

Upstream states that its current source code permits commercial use subject to
the VGGT License and Acceptable Use Policy. Checkpoint permissions are separate:
the original `facebook/VGGT-1B` checkpoint remains non-commercial, while the
gated `facebook/VGGT-1B-Commercial` checkpoint is the checkpoint upstream
designates for permitted commercial use under its applicable terms.

## Artifact decisions

| Artifact | `v0.1` decision | Required handling |
|---|---|---|
| Original VGGT source | Do not redistribute | Clone the pinned official repository when conversion-time inspection is required |
| VGGT-MLX source | Publish | Distribute under the included VGGT License; retain attribution, warranty, and Acceptable Use Policy terms |
| `facebook/VGGT-1B` original weights | Do not redistribute | Download from the official Hugging Face repository; clearly label use non-commercial |
| MLX weights converted from `VGGT-1B` | **Do not publish** | Keep local and gitignored; distribute deterministic conversion tooling only |
| `VGGT-1B-Commercial` original weights | Do not redistribute | Users obtain access directly through Meta's gated Hugging Face repository |
| MLX weights converted from `VGGT-1B-Commercial` | **Do not publish** | A future release requires separate written review of the gated terms and distribution mechanism |
| Conversion script and manifest format | Publish | Pin source/checkpoint revisions, retain notices, and never embed credentials or weights |
| Oracle `.npz` activation fixtures | Publish for current test release | Treat as model-derived test data under the repository's VGGT License; keep them limited to parity evidence |
| Generated depth and PLY demo outputs | Publish only with input rights | Attribute VGGT use and preserve any rights or consent required by the source images |
| Current sample photographs | **Blocked pending provenance** | Do not use in launch media or a public model card until source, license, and consent are recorded |
| Documentation, benchmark JSON, and original launch media | Publish | Credit VGGT and MLX; avoid implying endorsement or affiliation |

The no-weight decision is intentionally stricter than a claim that conversion
necessarily forbids redistribution. It avoids bypassing the official checkpoint
pages, access workflow, usage notice, and checkpoint-specific restrictions. A
future decision to upload converted weights must identify the exact source
checkpoint and revision, preserve all applicable terms, and be reviewed before
upload—not after.

## Required notices

Every public repository, package, model card, or substantial derived release
must:

1. identify the project as an **unofficial community port**;
2. credit the VGGT authors and link the official repository and paper;
3. include the VGGT License and Acceptable Use Policy wherever VGGT-derived code
   or artifacts are distributed;
4. distinguish source-code permission from checkpoint permission;
5. state that the original `VGGT-1B` checkpoint is non-commercial;
6. direct commercial users to obtain `VGGT-1B-Commercial` from Meta's gated
   repository and comply with its current terms;
7. state that inference outputs are machine-generated;
8. avoid Meta, VGGT, Apple, MLX, or Hugging Face logos and wording that imply
   sponsorship, approval, or official status.

Suggested short notice:

> Unofficial MLX port of VGGT. VGGT was created by its original authors at the
> Visual Geometry Group and Meta AI. This project is not affiliated with or
> endorsed by Meta, Apple, or the MLX team. VGGT source, checkpoints, and
> derivatives remain subject to their applicable terms; this repository does
> not distribute model weights.

## Citation

Technical articles, model cards, and research results must cite the original
work:

```bibtex
@inproceedings{wang2025vggt,
  title     = {VGGT: Visual Geometry Grounded Transformer},
  author    = {Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and
               Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and
               Pattern Recognition},
  year      = {2025}
}
```

Software citations should additionally name `vggt-mlx`, its release version,
repository URL, and release authors without replacing the upstream citation.

## Checkpoint acquisition policy

The setup flow must require an explicit checkpoint family. It must never
silently substitute the original and commercial checkpoints.

For the original checkpoint, the supported flow is:

1. show the source repository, non-commercial restriction, and applicable
   terms;
2. obtain the checkpoint directly from `facebook/VGGT-1B`;
3. verify the pinned source revision or record the selected revision;
4. convert locally;
5. save the source revision and source/converted checksums in the conversion
   report;
6. keep the converted artifact out of git and distribution packages.

Commercial-checkpoint conversion is out of scope for `v0.1`. Adding it requires
a separately tested adapter and a fresh review of the gated terms.

## Sample-image blocker

The committed images under `tests/fixtures/sample_images/` currently lack a
recorded source and license. Before making the repository public, do one of the
following:

- document their creator, source, license, and permission; or
- replace them and regenerate both oracle fixtures with an owned or explicitly
  redistributable three-view scene.

Until then, the images, their derived oracle fixtures, and the current demo GIF
must not be used as public launch assets. This blocker does not invalidate local
numerical testing, but it blocks public distribution of those artifacts.

## Pre-release audit

Run before every tag:

```bash
git ls-files | grep -E '\.(safetensors|pt|pth|ckpt)$' && exit 1 || true
git grep -nE 'hf_[A-Za-z0-9]{20,}' -- .
```

Then verify manually:

- `LICENSE.txt` matches the current authoritative upstream license;
- README, package metadata, and model card describe the same license boundary;
- no original or converted checkpoint is present in git history or artifacts;
- sample and demo assets have recorded publication rights;
- all citations and upstream links resolve;
- the release is described as unofficial and does not imply endorsement.

## Approval record

| Date | Scope | Decision |
|---|---|---|
| 2026-08-21 | `v0.1` source, conversion tooling, docs, tests, and benchmark evidence | Publish under included VGGT License, subject to the requirements above |
| 2026-08-21 | Original and commercial checkpoints and their MLX conversions | Do not redistribute; conversion-only workflow |
| 2026-08-21 | Existing sample images, oracle fixtures, and demo media | Public distribution blocked until provenance is resolved |
