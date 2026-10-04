---
license: apache-2.0
language:
  - he
  - en
tags:
  - ocr
  - hebrew
  - paddleocr
  - svtrv2
  - onnx
  - rtl
  - bidi
library_name: onnxruntime
pipeline_tag: image-to-text
---

> **Finetuned from PaddleOCR.** Every model in this repository is a finetune of a
> PaddleOCR model (PaddleOCR v3.7.0, Apache-2.0) and is released under the same
> licence — see the per-model `MODEL_CARD.md` for each lineage.
>
> **All reported CER figures are in-house evaluation** on our own test sets:
> measured, but vendor-reported and not independently verified. The benchmark
> harness is not shipped; methodology is documented in the
> [GitHub repo](https://github.com/RivoksLab/paddleocr-hebrew).
>
> **Training weights (`.pdparams`) and training configs are not published.** This is
> a release of inference-ready ONNX models, not a reproducible training pipeline.


# paddleocr-hebrew — models

**Built by [Rivok Labs](https://rivoklabs.com)** — [rivoklabs.com](https://rivoklabs.com) · [ronen@rivoklabs.com](mailto:ronen@rivoklabs.com) · [code on GitHub](https://github.com/RivoksLab/paddleocr-hebrew)

ONNX weights for **Hebrew OCR** (finetuned from PaddleOCR / SVTRv2). Runnable
pipeline code, quickstart, benchmark, and docs are on GitHub:
**https://github.com/RivoksLab/paddleocr-hebrew**.

All recognizers share one byte-identical 120-char charset
(`charset_v2f.txt`, md5 `e17ce22e7b4ab8224a3dad9e4c85b6ae`). Each folder has a
`MODEL_CARD.md` + `md5sums.txt`. Split-ONNX pairs (`nrtr-encoder` + `nrtr-decstep`)
are one logical model — the attention decode loop runs on the host.

> **Hebrew is RTL — output is LOGICAL Unicode order.** Apply `python-bidi`
> `get_display()` only when rendering, never before storing/scoring. This is the
> #1 way to get garbage out. See the GitHub `docs/charset.md`.

## Contents

| folder | role | files | size |
|---|---|---|---|
| `server-svtrv2/` | **flagship** server REC (SVTRv2, CTC + NRTR) | `ctc.onnx`, `nrtr-encoder.onnx`, `nrtr-decstep.onnx` | 77 + 72 + 27 MB |
| `light-svtrv2small/` | edge/CPU REC (NRTR-only KD student) | `nrtr-encoder.onnx`, `nrtr-decstep.onnx` | 28 + 27 MB |
| `server-v5/` | alt word-level server REC (PPHGNetV2-B4) | `rec.onnx` | 73 MB |
| `server-v6/` | alt word-level server REC (PPLCNetV4) | `rec.onnx` | 60 MB |
| `mobile-word/` | mobile word REC (PPLCNetV3 KD) | `rec.onnx` | 7.4 MB |
| `word-det/` | **word detector** (mobile DBNet) | `det.onnx` | 4.6 MB |
| `line-det/` | line detector (situational) | `det.onnx` | 4.6 MB |

## Quickstart

```bash
pip install "git+https://github.com/RivoksLab/paddleocr-hebrew" huggingface_hub
hf download Rivok/paddleocr-hebrew \
    --include "charset_v2f.txt" "word-det/*" "server-svtrv2/*" \
    --local-dir hebrew-ocr-models
```

```python
from ocr import HebrewOCR
ocr = HebrewOCR(models_dir="hebrew-ocr-models")
for line in ocr.read("page.png")["lines"]:
    print(line["text"])   # logical order
```

## Benchmark (headline)

- Clean pure-Hebrew (perfect GT): SVTRv2 NRTR **0.35%** CER vs Tesseract 1.34%.
- Real bilingual heb+lat (n=233): cascade/NRTR **2.33%** vs Tesseract 16.14%.
- Page-level (71-page GCV): word-DET + SVTRv2 **7.56%** vs Tesseract 14.20%.

Full tables, methodology, and the CTC-vs-attention finding: see GitHub.

## Fine-tuning

Released as **ONNX** only (inference-ready, portable — CPU/CUDA/Jetson). Paddle
training weights + configs are not published. To fine-tune on your own Hebrew
data or collaborate, open an issue on the GitHub repo or reach out:
**ronen@rivoklabs.com**.

## About Rivok Labs

[**Rivok Labs**](https://rivoklabs.com) builds computational intelligence and automation
tools, with a particular focus on Hebrew and other right-to-left languages that
mainstream tooling handles badly.

We released these models because there was no open-source Hebrew OCR with a
commercial-friendly licence that held up on real documents. If they are useful to you,
or if they fail on your documents, we would like to hear about it.

- **Web:** [rivoklabs.com](https://rivoklabs.com)
- **Contact:** [ronen@rivoklabs.com](mailto:ronen@rivoklabs.com)
- **Issues and discussions:** [github.com/RivoksLab/paddleocr-hebrew](https://github.com/RivoksLab/paddleocr-hebrew)

## License

Apache-2.0. Finetuned from PaddleOCR (Apache-2.0) — see GitHub `NOTICE`.
