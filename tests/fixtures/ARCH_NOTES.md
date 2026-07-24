# VGGT Architecture Notes

Verified against the canonical upstream `facebookresearch/vggt` source on
2026-07-24. Re-run `tools/introspect.py` with `facebook/VGGT-1B` to regenerate
this file together with the checkpoint state-dict inventory and module repr.

## Verified aggregator literals

- `aa_order`: `['frame', 'global']`
- `aa_block_size`: `1`
- `rope_freq`: `100`
- `qk_norm`: `True`
- LayerScale `init_values`: `0.01`
- `camera_token` shape: `(1, 2, 1, 1024)`
- `register_token` shape: `(1, 2, 4, 1024)`
- frame block container: `frame_blocks`
- global block container: `global_blocks`
- `patch_start_idx`: `5`

## Special-token behavior

Both learned token tensors have two slots. Slot 0 is used for the reference
(first) frame. Slot 1 is shared by every remaining frame. Each frame receives
one camera token followed by four register tokens, so patch tokens begin at
index 5.

## Backbone and alternating-attention structure

The default `patch_embed="dinov2_vitl14_reg"` is the complete DINOv2
ViT-L/14-with-registers module, including its 24 transformer blocks. Its
`x_norm_patchtokens` output is then consumed by 24 `frame_blocks` and 24
`global_blocks`, alternating one block at a time. Cached frame and global
outputs are concatenated on the feature axis, producing the 2048-dimensional
head input at layers 4, 11, 17, and 23.
