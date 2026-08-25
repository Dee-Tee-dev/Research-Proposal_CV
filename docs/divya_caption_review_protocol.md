# Divya Caption-Review Protocol

Divya's share is the even-positioned half of the 168 sorted image IDs: 84
images, with baseline and prompted BLIP captions for each image. The split is
deterministic and does not select examples based on whether the model was
correct.

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
Instead, clear metric false negatives and ambiguous cases are reported as an
audit of the automatic recall measure. This preserves the preregistered metric
while documenting its limitations.
