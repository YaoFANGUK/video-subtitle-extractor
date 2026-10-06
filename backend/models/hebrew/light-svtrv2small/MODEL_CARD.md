# light-svtrv2small — Hebrew SVTRv2-small REC (edge / x86-CPU tier)

**Role:** Lightweight single-pass Hebrew recognizer for edge / x86-CPU deployment
(~4× faster encoder than the server SVTRv2 teacher). = "SVTRv2-small bilingual KD ep8".

## Architecture
- Backbone **SVTRv2-small** (dims `[64,128,256]`, heads `[2,4,8]`), NRTR head
  (`nrtr_dim 384`, 2 decoder layers).
- **Ships NRTR-only.** The light-CTC quadrant was tested and failed — the small
  backbone cannot produce cleanly CTC-alignable Hebrew features (CTC heb_only ~54%),
  while its NRTR head reads Hebrew at 3.5%. So this tier is NRTR-only.
- Deployed like the server NRTR: **split ONNX** (`nrtr-encoder.onnx` → memory,
  `nrtr-decstep.onnx` → one step) + numpy host greedy loop (paddle-free; Jetson
  onnxruntime-gpu or x86 CPU). Full-loop ONNX SIGABRTs on control flow → split is required.

## Provenance
Same-family **knowledge distillation** from the server SVTRv2 ft2 ep7 teacher:
teacher → ① longtext-KD foundation ep16 → **Stage-2 bilingual KD ep8** (this model).
Validated: Jetson host-loop ONNX ≈ paddle-native (heb+lat 3.77 vs 3.81, heb+dig 1.85 = 1.85).

## Files
- `nrtr-encoder.onnx` (28 MB) + `nrtr-decstep.onnx` (27 MB) — split NRTR, host-loop driven.
- md5s in `md5sums.txt`.

## Charset
All rec models in this release share the same **120-char Hebrew charset**
(`charset_v2f.txt` at the repo root). The CTC vocabulary is
`["blank"] + <120 chars> + [" "]` (space appended as the last index; `use_space_char=true`).

## Eval — NRTR micro-CER by slice
| slice | this (ep8) | teacher ceiling |
|---|---:|---:|
| heb_only (n=919) | 3.51% | 0.36% |
| heb+lat (n=233) | 3.81% | 2.37% |
| heb+dig (n=108) | 1.85% | 0.56% |
| lat_only (n=92) | 2.67% | ~1.1% |
| dig_only (n=138) | 1.97% | ~1.4% |

Beats its ① foundation on every core slice. Strong for a small backbone.

## Known limitations
- **Capacity-limited vs the teacher** (~1.5–1.6× the teacher's mixed-slice CER) —
  expected for the small backbone. Use the server SVTRv2 flagship where accuracy matters more
  than speed/size.
- NRTR-only: no fast single-pass CTC path in this tier.

## License
Apache-2.0. Derivative of PaddleOCR (Apache-2.0).
