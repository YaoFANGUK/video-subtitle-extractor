# server-v5 — Hebrew PP-OCRv5 word-level server REC

**Role:** Alternative word-level server recognizer (the prior production server teacher).
= "step1_with_spaces **v2.1 ep4**". Word-level (`max_text_length=25`, `use_space_char=true`).

## Architecture
- Backbone **PPHGNetV2_B4** (PP-OCRv5 server rec), **CTC**. CTC-only ONNX.

## Provenance
The step1_with_spaces line of finetunes (retrained with `use_space_char=true`),
warm-started v1 → v2 → **v2.1 ep4**, the locked long-term server teacher. Also served as
the teacher for the mobile KD student (`mobile-word`) and the SVTRv2 line-rec lineage.

## Files
- `rec.onnx` (73 MB) — CTC head, ONNX runtime (Jetson Docker + CUDA, x86 ORT).
- md5 in `md5sums.txt`.

## Charset
All rec models in this release share the same **120-char Hebrew charset**
(`charset_v2f.txt` at the repo root). The CTC vocabulary is
`["blank"] + <120 chars> + [" "]` (space appended as the last index; `use_space_char=true`).

## Eval — micro-CER
- prod-weighted CER **1.523%**
- heb_only **1.26%**
- (word-level slices, ≤25 chars; BiDi-normalized dyn16.)

## Role & usage
Word-DET + single-pass word recognizer. For **long lines** it needs the external
sliding-window inference technique (v2.1 ep4 + windows w=320/o=200, 3-way ensemble),
unlike SVTRv2 which reads long lines natively. Prefer `server-svtrv2` for long-line /
inline-bilingual text; this model is a compact CTC alternative for the word-DET path.

## Known limitations
- Word-level only: caps at 25 chars; long single-token inputs are out-of-distribution
  without the sliding-window wrapper.
- Weaker than SVTRv2 on embedded Latin-in-Hebrew (~16% heb+lat on the mixed slice).

## License
Apache-2.0. Derivative of PaddleOCR (Apache-2.0).
