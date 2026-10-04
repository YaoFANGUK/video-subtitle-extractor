# server-svtrv2 — Hebrew SVTRv2 REC (FLAGSHIP recognizer)

**Role:** Flagship Hebrew text-recognition model. Reads long lines and mixed
Hebrew/Latin/digit script natively, single-pass (no sliding window). = "SVTRv2
bilingual **ft2 ep7**".

## Architecture
- Backbone **SVTRv2** (algo `SVTR_HGNet`, PaddleOCR v3.7.0), **MultiHead: CTC + NRTR**,
  sharing one backbone.
- Two deploy heads:
  1. **CTC** — fast single-pass. `ctc.onnx`.
  2. **NRTR** — quality head for mixed script. Deployed as a **split ONNX**
     (`nrtr-encoder.onnx` → memory, `nrtr-decstep.onnx` → one autoregressive step)
     driven by a host-side numpy greedy loop. This split is **required, not optional**:
     a full-loop NRTR ONNX export SIGABRTs on control flow (`while`/`if`). The
     encoder + per-step decoder + host loop reproduces paddle exactly.

## Provenance
Chinese SVTRv2 pretrain → Hebrew longtext finetune → **two bilingual finetune rounds**
(ft → ft2). ft2 ep7 warm-started from ft ep7 on font-stack realistic bilingual synth
to close the heb+lat residual; best of an 8-epoch run, zero pure-script regression.
Paddle `.pdparams` → ONNX via ABI-matched paddle 3.2.2 + paddle2onnx 2.1.0.

## Files
- `ctc.onnx` (77 MB) — CTC head, single-pass fast path.
- `nrtr-encoder.onnx` (72 MB) + `nrtr-decstep.onnx` (27 MB) — split NRTR, host-loop driven.
- md5s in `md5sums.txt`.

## Charset
All rec models in this release share the same **120-char Hebrew charset**
(`charset_v2f.txt` at the repo root). The CTC vocabulary is
`["blank"] + <120 chars> + [" "]` (space appended as the last index; `use_space_char=true`).

## Eval — micro-CER by slice (dyn16 / max_w 1280)
| slice | CTC head | NRTR head |
|---|---:|---:|
| heb_only (pure) | 0.64% | 0.36% |
| lat_only | 1.20% | ~1.1% |
| dig_only | 1.41% | ~1.4% |
| pure long Hebrew (71–80 char) | **0.49%** | — |
| heb+lat (embedded Latin, n=233) | 12.63% | **2.33%** |
| heb+dig (embedded digits, n=108) | 4.99% | **0.56%** |

Pure long Hebrew at 0.49% breaks the old ~52% line-level wall. The mixed-script split
is the key story: embedded LTR runs (Latin/digit inside RTL Hebrew) — **CTC deletes them,
NRTR reads them.**

## Recommended deployment — script-gated cascade
Default to CTC; fall back to the NRTR split-ONNX when the CTC output contains a
digit/Latin character or CTC confidence is low. This hits the NRTR ceiling on mixed
slices at ~10% fallback rate / ~1.25–1.36× CTC latency.

## Known limitations
- The **CTC head alone is weak on embedded LTR islands** (deletes Latin/digit runs) —
  use the cascade or the NRTR head for bilingual/inline-numeric text.
- **geresh/gershayim (`'`/`"`) can be misread as yod (`י`)** in page-ref contexts
  (`עמ'`, `מס'`, `ס"ק`) — fixable by the shipped deterministic post-processor.

## License
Apache-2.0. Derivative of PaddleOCR (Apache-2.0).
