# Dollar Street Balanced Subset - Data Quality Report

## Dataset and grain

- Source grain: one image record.
- Audited Hugging Face viewer rows: 1600.
- Final manifest rows: 168 unique image assignments.
- Intended experimental grain: one image x one study category x one income quartile.

## Validation results

- Automated checks passed: 11/11.
- Duplicate manifest IDs: 0.
- Missing required manifest values: 0.
- Category-quartile balance: 7 images in every one of the 24 cells.
- Countries represented in the manifest: 44.

## Source-data findings

- 45 of 1600 source rows (2.8%) contain more than one ImageNet class and were excluded from category selection.
- The repository CSV contains 3,616 rows, while the current Hugging Face viewer/parquet split exposes its first 1,600 rows. The manifest is based on the exposed 1,600-row split so it matches the planned pipeline.

## Regional distribution

| Region code | Images | Share |
|---|---:|---:|
| af | 39 | 23.2% |
| am | 44 | 26.2% |
| as | 64 | 38.1% |
| eu | 21 | 12.5% |

## Risk assessment

- **Income-quartile analysis: low data-quality risk.** The design is exactly balanced by category and quartile, with unique images and complete required metadata.
- **Regional analysis: medium risk.** All four regions are represented, but the manifest is not region-balanced; Asia has the largest share and Europe the smallest. Regional findings must remain secondary and include sample counts.
- **Caption metric: medium risk.** Source ImageNet names can be culturally narrow. The manifest therefore stores broader accepted caption terms, but automated term matching must still be paired with the planned blinded manual audit.

## Stable automated tests

- Require exactly 168 unique image IDs.
- Require exactly seven rows per category-income-quartile cell.
- Reject missing category, income, region, country, image name, or accepted-term fields.
- Reject manifest IDs absent from the normalized 1,600-row metadata table.
- Treat region-level metrics as invalid unless every reported group includes its sample count.

Overall validation status: **PASS**.
