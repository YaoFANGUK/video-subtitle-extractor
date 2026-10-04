# Candidate Subtitle Frames

The log messages `Extracting candidate subtitle frames...` and
`Candidate subtitle frames extracted` (`Main.StartProcessFrame` /
`FinishProcessFrame` in `backend/interface/*.ini`) refer to the frames that are
queued for OCR. They were previously worded "video keyframes", which was
misleading: **codec keyframes (I-frames) are never used** to choose frames.

A *candidate frame* is a frame that is likely to carry subtitle text and is
handed to the OCR stage (`subtitle_ocr_task_queue`). The selection differs per
mode. All paths are in `backend/main.py`.

## Accurate mode with a selected area: `extract_frame_by_det`

Used when a subtitle area is selected and a hardware accelerator is available.

1. Every frame is decoded sequentially (no sampling).
2. The detector (`SubtitleDetector.detect_subtitle`, ROI-cropped, see
   [ocr-roi.md](ocr-roi.md)) finds text boxes. A frame `has_subtitle` when at
   least one box lies fully inside the selected area.
3. A small state machine finds subtitle boundaries:
   - **Start**: first frame with a subtitle after none. It is OCR'd
     immediately and cached as the reference text.
   - **End**: while tracking, each frame with a subtitle is OCR'd and compared
     with the start frame's text (`_compare_ocr_result`, Levenshtein
     `ratio` against `thresholdTextSimilarity`). The subtitle ends at the frame
     before the text stops matching, or before the subtitle disappears, or at
     the last video frame.
4. Only the start frame and the end frame (the last frame showing the same
   text) of each subtitle are queued, with their already computed `dt_box` /
   `rec_res`. The OCR stage turns these into start/end timestamps and text.

So the candidates are the **boundary frames of each subtitle**, not a sample.
Cost is one detection per frame plus a few OCR calls per subtitle.

## Fast mode (or no accelerator) with an area: `extract_frame_by_vsf`

[VideoSubFinder](https://sourceforge.net/projects/videosubfinder/) (VSF) is a
bundled third-party binary (`backend/subfinder/<os>/`). Its source is not in
this repository, so the **frame-selection algorithm itself is VSF's and is not
verifiable here**; what follows separates what this repository does (verified
from `extract_frame_by_vsf`) from VSF's general approach (from VSF's public
description, not checked against its code).

### What this project does (verified)

1. The selected area is converted to fractions of the frame and passed as
   `-te/-be/-le/-re` (top/bottom/left/right edge), so VSF only inspects that
   band. Other flags: `-c` clear the output folder, `-r` run the search,
   `-ces` create an empty subtitle file, `-nthr` worker threads (CPU count,
   or `videoSubFinderCpuCores`), `--use_cuda` when available, and
   `--open_video_<decoder>` (configured decoder).
2. VSF runs as a child process and writes the frames it selects into
   `<temp>/RGBImages/` named `h_mm_ss_ms__<end time>__<id>.jpeg`.
3. A watcher thread (`count_process`; on non-Windows `vsf_output`, which parses
   VSF's stderr `Frame:` lines) polls for new files, converts the leading
   timestamp to milliseconds, and queues each new one as an OCR task
   `(frame_count, frame_no, None, None, total_ms, subtitle_area)`. Timestamps
   that do not increase are skipped, and progress is `total_ms / duration_ms`.
4. The OCR stage then reads those images, recognises text, and merges
   neighbouring results with similar text into one subtitle (Levenshtein
   `ratio` against `thresholdTextSimilarity`).

### How VSF selects frames (general approach, unverified)

VSF's first stage ("search for frames with subtitles") works on the area band
only, and does not depend on codec structure:

1. Decode frames in sequence (it can skip/sample to speed up, which is why
   very short subtitles can be missed).
2. Apply a text-likeness filter to the band (colour/edge filtering that keeps
   high-contrast, thin strokes typical of text) producing a binary mask.
3. Compare the mask with the previous frame's mask. A sustained, mostly
   unchanged mask means a subtitle is on screen; a clear change means the
   subtitle started, ended or switched text.
4. For each detected subtitle interval it writes representative image(s) and
   the interval's start/end times in the file name.

Practical consequences: output frames carry subtitle timing, one subtitle may
yield several images (deduplicated later by text similarity), and quality
depends on the area being tight and on contrast against the background.
Confirming the exact thresholds would require VSF's source or its own docs.

## No subtitle area: `extract_frame_by_fps`

Fixed-interval sampling: every `fps // extractFrequency` frames are queued.
Fast, but short subtitles can be missed if the interval is too large.

## Codec keyframes

The only place keyframes appear in the repository is the bundled Sushi
subtitle-synchronisation tool (`backend/sushi`), which is unrelated to frame
extraction.

## Wording change

All eight language files (`en`, `ch`, `chinese_cht`, `japan`, `ko`, `es`, `tr`,
`vi`) now say "candidate subtitle frames" instead of "(key)frames". The keys
are unchanged, so no code depends on the new text.
