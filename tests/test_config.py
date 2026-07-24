from vggt_mlx.config import VGGTConfig


def test_config_matches_verified_architecture_notes():
    config = VGGTConfig()
    assert (config.img_size, config.patch_size, config.embed_dim) == (518, 14, 1024)
    assert (config.depth, config.num_heads, config.mlp_ratio) == (24, 16, 4.0)
    assert (config.aa_order, config.aa_block_size, config.qk_norm) == (
        ("frame", "global"),
        1,
        True,
    )
    assert (config.rope_freq, config.layerscale_init) == (100, 0.01)
    assert (config.num_register_tokens, config.patch_start_idx) == (4, 5)


def test_head_defaults():
    config = VGGTConfig()
    assert config.intermediate_layer_idx == (4, 11, 17, 23)
    assert (config.dpt_features, config.dpt_out_channels) == (
        256,
        (256, 512, 1024, 1024),
    )
    assert (config.pose_encoding_type, config.pose_dim) == ("absT_quaR_FoV", 9)
