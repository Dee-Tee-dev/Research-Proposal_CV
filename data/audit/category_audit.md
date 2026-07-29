# Dollar Street Test Metadata Audit

- Dataset rows: 1600
- Rows with one unique ImageNet class, numeric income, and region: 1555
- Income quartile cutpoints: Q1 <= 210.67, Q2 <= 685.00, Q3 <= 1841.00, Q4 above 1841.00
- Unique eligible ImageNet classes: 58

## Final recommended controlled subset

Use 7 images per category per income quartile: 6 categories x 4 quartiles x 7 = 168 images.

| Study label | Source class | Q1 | Q2 | Q3 | Q4 | Accepted caption terms |
|---|---|---:|---:|---:|---:|---|
| roof | tile roof (858) | 14 | 14 | 12 | 12 | roof, roofing, ceiling |
| light source | table lamp (846) | 22 | 12 | 17 | 22 | light, lamp, bulb, lantern, candle |
| stove | stove (827) | 8 | 8 | 14 | 12 | stove, oven, cooker, cooktop, hob |
| trash container | trash can (412) | 12 | 7 | 8 | 15 | trash can, garbage can, garbage bin, waste bin, dustbin, dumpster |
| switch | switch (844) | 7 | 11 | 10 | 13 | switch, light switch, power switch |
| footwear | running shoe (770) | 12 | 11 | 7 | 9 | shoe, shoes, sandal, sandals, slipper, slippers, sneaker, sneakers, boot, boots, footwear |

These six classes retain all four regions, have at least seven eligible images in every income quartile, and have interpretable visual concepts. Broader study labels and caption terms reduce false penalties caused by Western-specific ImageNet names such as `table lamp` and `running shoe`.

## Recommended categories

| Class | Synset | Total | Q1 | Q2 | Q3 | Q4 | Balanced n | Regions with >=3 | Countries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| day bed | 831 | 74 | 15 | 22 | 16 | 21 | 60 | 4 | 37 |
| table lamp | 846 | 73 | 22 | 12 | 17 | 22 | 48 | 4 | 27 |
| tile roof | 858 | 52 | 14 | 14 | 12 | 12 | 48 | 4 | 30 |
| dining table | 532 | 49 | 13 | 12 | 12 | 12 | 48 | 4 | 28 |
| soap dispenser | 804 | 55 | 16 | 19 | 9 | 11 | 36 | 4 | 30 |
| plate | 923 | 44 | 11 | 9 | 14 | 10 | 36 | 4 | 25 |

Selection rule: prefer classes with the largest minimum income-quartile cell, then broader regional coverage, then larger total count. `Balanced n` is four times the smallest quartile count and is the maximum equal-per-quartile sample.

## Top 20 category audit

| Class | Total | Minimum quartile | Balanced n | Region coverage |
|---|---:|---:|---:|---:|
| day bed | 74 | 15 | 60 | 4 |
| table lamp | 73 | 12 | 48 | 4 |
| tile roof | 52 | 12 | 48 | 4 |
| dining table | 49 | 12 | 48 | 4 |
| soap dispenser | 55 | 9 | 36 | 4 |
| plate | 44 | 9 | 36 | 4 |
| street sign | 54 | 8 | 32 | 4 |
| stove | 42 | 8 | 32 | 4 |
| salt shaker | 38 | 8 | 32 | 4 |
| trash can | 42 | 7 | 28 | 4 |
| switch | 41 | 7 | 28 | 4 |
| running shoe | 39 | 7 | 28 | 4 |
| bookcase | 35 | 7 | 28 | 4 |
| wardrobe | 36 | 5 | 20 | 4 |
| skillet | 35 | 5 | 20 | 4 |
| manufactured home | 29 | 5 | 20 | 4 |
| washbasin | 28 | 5 | 20 | 4 |
| wooden spoon | 26 | 5 | 20 | 4 |
| soda bottle | 25 | 5 | 20 | 3 |
| plate rack | 20 | 5 | 20 | 3 |
