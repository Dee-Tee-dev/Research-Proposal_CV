# Divya Caption-Review Protocol

Divya's share is the even-positioned half of the 168 sorted image IDs: 84
images, with baseline BLIP, prompted BLIP, and Qwen captions for each selected
image. The split is deterministic and does not select examples based on whether
the model was correct. The two BLIP conditions have already been reviewed. The
same 84 image IDs and decision rules are used for the separate Qwen review once
the full Qwen run is complete.

Each caption receives one semantic target decision:

- `yes`: the caption identifies the study object or a clearly correct synonym;
- `no`: it identifies another object or misses the study object;
- `uncertain`: the image, source label, or wording does not support a confident
  binary judgment.

Caption quality is separately marked `accurate`, `inaccurate`, `ambiguous`, or
`disfluent`. This prevents a correct object word in a badly formed caption from
being treated as fully satisfactory. Automatic accepted-term matches are
checked in context, while automatic misses are inspected against the image.

The review does not retroactively change the predefined accepted-term list.
Instead, clear metric false negatives and ambiguous cases are reported by model
as an audit of the automatic recall measure. This preserves the preregistered
metric while documenting its limitations. BLIP and Qwen decisions are stored
in separate completed files so new outputs cannot overwrite earlier judgments.

For Qwen, `scripts/create_qwen_review_sheets.py` produces 21 numbered sheets
with four records per page. Each record displays the original image, target
category, income quartile, generated caption, automatic-match result, and the
three decision fields. The sheets follow sorted image-ID order, matching the
CSV queue exactly, so every decision remains traceable to its prediction row.
After all decisions are recorded, `scripts/summarize_qwen_review.py` rejects
blank or invalid fields and produces the automatic recall, strict semantic
recall, uncertainty upper bound, clear false-negative and false-positive
counts, and disfluency count used in the final results section.
