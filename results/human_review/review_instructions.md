# Blinded BLIP Caption Review

Each reviewer must complete their CSV independently before discussing any rows.
Do not open `unblinding_key.csv`: it contains the income group and prompt condition.

Open the file named in `image_file`, compare it with `caption`, and enter only `1`
or `0` in all three rating columns:

- `object_correct`: 1 when the caption correctly identifies or clearly describes
  the main household object visible in the image; otherwise 0. Culturally valid
  everyday synonyms are acceptable.
- `non_informative`: 1 when the caption is too generic to identify the main object
  (for example, only "an indoor scene"); otherwise 0.
- `hallucination`: 1 when the caption makes a concrete claim about an object,
  attribute, or action that is clearly absent from or contradicted by the image;
  otherwise 0.

Use `reviewer_notes` only for uncertain cases. Do not change the review ID, image
file, caption, row order, or column names. After both completed files are locked,
run the agreement script and adjudicate only the listed disagreements.
