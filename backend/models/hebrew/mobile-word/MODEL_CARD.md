# mobile-word — Hebrew mobile word-level REC

**Role:** Mobile / edge word-level recognizer (7.4 MB). = "step2_kd_v2 ep18".
Word-level (≤25 chars). CTC-only.

## Architecture
- Backbone **PPLCNetV3** (PP-OCRv5 mobile rec), **CTC**. Knowledge-distilled student.

## Provenance
**Knowledge distillation** from the v5 server teacher (`server-v5` / step1_with_spaces
v2.1 ep4). Resumed KD run, ep18 selected as the ship epoch (validated best on heb_only
exact, digits, and Latin over the ep12–ep30 trajectory).

## Files
- `rec.onnx` (7.4 MB) — CTC head, ONNX runtime (Jetson Docker + CUDA, x86 ORT).
- md5 in `md5sums.txt`.

## Charset
All rec models in this release share the same **120-char Hebrew charset**
(`charset_v2f.txt` at the repo root). The CTC vocabulary is
`["blank"] + <120 chars> + [" "]` (space appended as the last index; `use_space_char=true`).

## Eval — word-level slices (BiDi-normalized)
| slice | macro-CER | exact |
|---|---:|---:|
| heb_only | 12.53% | 65.35% |
| lat_only | 4.00% | — |
| dig_only | 1.69% | — |

Clean across-the-board upgrade over the prior shipping mobile rec (step2_kd ep15).

## Role & usage
Mobile / edge word recognizer for the word-DET path. Smallest recognizer in the release
(~10–15× smaller than the server recs). Pair with `word-det`. Often used as the primary
in a two-pass cascade with a server rec as low-confidence fallback.

## Known limitations
- Word-level only (≤25 chars); no long-line capability.
- Higher CER than the server recs, especially on Hebrew — the size/speed trade-off.

## License
Apache-2.0. Derivative of PaddleOCR (Apache-2.0).
