# Dataset files

This directory contains metadata and a fixed experimental split, not a copied
version of the full Dollar Street image archive.

## Included

- `metadata/dollar_street_test_metadata.csv`: normalized metadata for the
  1,600 rows exposed by the current Hugging Face viewer/parquet split.
- `splits/balanced_subset_manifest.csv`: the 168-image controlled subset.
- `audit/`: selection evidence and data-quality checks.

## Images

Run:

```bash
python scripts/download_images.py
```

The downloader uses each manifest row's hosted dataset row index and stores the
image under `data/images/<image_name>`. This directory is ignored by Git.

Do not commit the images, the original zip archive, or model weights. Preserve
the original dataset attribution and licence information in any shared demo or
report.

