# line-det — Hebrew line-level text detector (situational)

**Role:** Line-level detector (one box per baseline row). Hebrew-finetuned mobile DBNet.
A **situational** alternative to `word-det` for degraded scans / hard layouts.

## Architecture
- **DBNet**, backbone **PPLCNetV3 (scale 0.75)** + **RSEFPN** + **DBHead** — same
  architecture as `word-det`, but trained on **line-level GT** (word boxes grouped into
  baseline rows). Mobile-class (~4.6 MB).

## Provenance
Same mobile DBNet + same warm-start as the word detector; only the ground truth differs
(line polygons instead of word boxes). Best epoch 18. Zero eval leakage vs the page-level test.

## Files
- `det.onnx` (4.6 MB) — ONNX runtime (Jetson Docker + CUDA, x86 ORT).
- md5 in `md5sums.txt`.

## Eval
- **val hmean 0.913** (best epoch 18).
- **Fair page-level test (71-page GCV-anchored, SVTRv2 cascade held constant):**
  word-DET **7.56%** vs this line-DET **10.36%** vs stock line-DET 10.67% vs Tesseract 14.20%.

## Role & usage — honest guidance
On clean pages, **`word-det` is the default and wins** the fair page-level comparison.
Use `line-det` only as a **candidate route for degraded scans / hard layouts**, where it
helps: on degraded `pd_phase1` it wins (28.8% vs word-DET 35.7%) and it aids some bilingual
cases. Pair with `server-svtrv2` single-pass (line-level reader — no seams).

## Known limitations
- Loses to `word-det` on typical clean Hebrew pages (see eval).
- One box per row → recognizer sees full lines; requires a line-capable rec
  (`server-svtrv2`), not the word-level recs.

## License
Apache-2.0. Derivative of PaddleOCR (Apache-2.0).
