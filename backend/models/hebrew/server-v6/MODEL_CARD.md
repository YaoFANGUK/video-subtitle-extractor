# server-v6 — Hebrew PP-OCRv6 word-level server REC

**Role:** Alternative word-level server recognizer. = "PP-OCRv6 Hebrew **v6.1 ep11**".
Word-level (`max_text_length=25`, `image_shape [3,48,320]`, `use_space_char=true`).
Beats the v5 word-level teacher on every script slice.

## Architecture
- Backbone **PPLCNetV4-medium** (PP-OCRv6, PaddleOCR v3.7.0), MultiHead (CTC + NRTR),
  **CTC-only at inference**.

## Provenance
- Phase 1: warm-start v6_medium pretrain → re-init heads on the Hebrew charset, trained on
  frozen `step1_with_spaces_v2_1` (211k), 16ep (pure-script gate passed, mixed regressed).
- Phase 2: warm-start ep16 on bilingual-boosted v2.2 data (de-leaked), 12ep → **ep11 ship**.
- Paddle `.pdparams` → ONNX via ABI-matched paddle 3.2.2 + paddle2onnx 2.1.0
  (validated <0.05pt vs paddle-native).

## Files
- `rec.onnx` (60 MB) — CTC head, ONNX runtime (Jetson Docker + CUDA, x86 ORT).
- md5 in `md5sums.txt`.

## Charset
All rec models in this release share the same **120-char Hebrew charset**
(`charset_v2f.txt` at the repo root). The CTC vocabulary is
`["blank"] + <120 chars> + [" "]` (space appended as the last index; `use_space_char=true`).

## Eval — macro-CER / exact (word-level slices, ≤25 chars, BiDi-normalized)
| slice | v5 v2.1 ep4 | **v6.1 ep11** |
|---|---|---|
| heb_only | 2.80% / 89.58% | **2.17% / 90.88%** |
| lat_only | 2.32% / 93.30% | **1.99% / 94.92%** |
| dig_only (despace) | ~2.0% | **0.62%** |
| heb+lat (n=112) | 27.02% / 10.71% | **18.83% / 30.36%** |
| heb+dig (n=183) | 15.32% / 32.79% | **14.86% / 43.72%** |

## Known limitations
- **⚠️ v6 (any size) CANNOT do sliding-window long-line inference** — architecturally
  verified. It ships as a **short-crop / word-DET rec only**. For long lines use
  `server-svtrv2` (the long-line tool); the v5 sliding-window trick does not transfer to v6.
- ≥16-char length cliff (word-model horizon).
- geresh/gershayim ↔ yod in page-ref contexts (`עמ'`/`מס' NN`).
- Latin-inside-Hebrew is the weakest slice (~45% of heb+lat errors are Latin-side).

## License
Apache-2.0. Derivative of PaddleOCR (Apache-2.0).
