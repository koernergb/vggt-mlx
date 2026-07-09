def test_scaffold_imports():
    import vggt_mlx
    import vggt_mlx.heads.dpt_head
    import vggt_mlx.layers.attention
    import vggt_mlx.models.vggt

    assert vggt_mlx.__version__ == "0.0.0"
