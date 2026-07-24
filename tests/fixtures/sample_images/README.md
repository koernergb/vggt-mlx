# Oracle sample images

Place three related RGB photographs here before running Card 0.3. Keep the
images small enough to commit and use stable filenames because Card 5.1 will
re-run preprocessing against these exact files.

Generate both fixtures from the repository root with:

```bash
python tools/oracle.py \
  tests/fixtures/sample_images/view_0.jpg \
  tests/fixtures/sample_images/view_1.jpg \
  tests/fixtures/sample_images/view_2.jpg
```

Checkpoint weights remain ignored. The two generated oracle `.npz` fixtures
are explicitly allowed by `.gitignore` because they are parity test data.
