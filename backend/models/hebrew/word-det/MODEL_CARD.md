# word-det — Hebrew word-level text detector

**Role:** Word-level detector (one box per word) for the flagship page pipeline.
Hebrew-finetuned mobile DBNet. = "phase1_v2b_mild".

## Architecture
- **DBNet** (Differentiable Binarization), backbone **PPLCNetV3 (scale 0.75)** + **RSEFPN**
  neck + **DBHead**. Mobile-class (~4.6 MB).

## Provenance
Warm-started from a PP-OCR mobile DBNet and finetuned on Hebrew document word-box GT.

## Files
- `det.onnx` (4.6 MB) — ONNX runtime (Jetson Docker + CUDA, x86 ORT).
- md5 in `md5sums.txt`.

## Eval
- **hmean 0.8415** at det inference `max_side=1280`.
- **Flagship page pipeline:** word-DET + SVTRv2 single-pass = **7.56% page-CER** on the
  71-page GCV-anchored eval — beats line-DET (10.36%) and Tesseract 5 (14.20%). The word-DET
  advantage over line-DET is a genuine detection-quality difference, confirmed on a fair
  (Hebrew-finetuned) line-DET comparison.

## Recommended detection inference
- `max_side = 1280`
- `det_db_box_thresh ≈ 0.3–0.4`
- `det_db_unclip_ratio ≈ 1.3`

## Role & usage
Default detector for Hebrew pages. Pair with a rec model (flagship: `server-svtrv2`
single-pass). Produces word crops (≤25 chars typical), so it also suits the word-level
recognizers (`server-v5`, `server-v6`, `mobile-word`).

## Known limitations
- Over-fragments **degraded/thermal print** (splits smeared words into fragments) — for
  degraded scans / hard layouts consider `line-det` instead.

## License
Apache-2.0. Derivative of PaddleOCR (Apache-2.0).
