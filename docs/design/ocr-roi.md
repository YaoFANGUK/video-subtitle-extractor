# Selected-Area OCR Region of Interest

## Status

The `SubtitleROI` geometry helper, `SubtitleArea.roi` ownership, and ROI-aware detection/OCR paths are implemented. A selected area's ROI is initialized when extraction starts, then passed through accurate-mode detection/comparison and asynchronous OCR. Detector and OCR methods crop internally and return full-frame coordinates. Unit and boundary tests pass; comparison on a representative real video remains.

## Goals

- Avoid running text detection and recognition over frame regions that cannot contribute to selected subtitle areas.
- Preserve current selection, overflow-tolerance, confidence-filtering, cache, and output behavior as much as possible.
- Keep all coordinates outside a model crop in the existing full-frame coordinate system.
- Compute each area's crop geometry and coordinate transforms once per video, then reuse them for every frame.
- Prepare the data model for multiple selected areas, with one ROI per area.
- Fail clearly if frame dimensions differ from the dimensions used to construct an ROI.

## Current behavior

With a user-selected subtitle area, accurate-mode detection and comparison OCR in `backend/main.py` receive full frames. The asynchronous OCR worker in `backend/tools/subtitle_ocr.py` also receives full frames, then filters recognized boxes against the selected area. ROI-aware detector/OCR methods now crop internally when an ROI is supplied and translate results back to full-frame coordinates. Without a custom selection, the prior full-frame behavior remains.

The separate upper/lower-half crop (`frame_preprocess`) is a coarse preprocessing option. For a custom selected area, its precise ROI crop takes precedence; the coarse half-frame crop is only applied when no custom ROI is present.

The UI can represent multiple selected areas, but current task plumbing passes only the first area to extraction. Multi-area extraction is a future phase; ROI ownership should already be per-area so each selection has independent geometry.

## Design decisions

### One ROI per selected area and video

Each selected `SubtitleArea` owns its own `SubtitleROI`. The ROI is created from that area's coordinates, the video's frame dimensions, and context padding. Its clipped crop bounds, crop origin, crop-to-frame offset, and frame-to-crop offset are calculated once. Pixel data changes per frame, so each frame is cropped using the stored geometry.

`SubtitleArea` objects are created in the GUI before video dimensions are available, so their constructors cannot initialize an ROI from selection coordinates alone. The per-video extractor calls `initialize_roi()` once it knows the frame dimensions. When multi-area processing is implemented, it must initialize every selected area's ROI independently; an ROI must never be shared between different selections.

This assumes video frames have stable dimensions. `SubtitleROI.crop()` checks the dimensions and raises an error rather than silently using stale bounds if they change.

### Context padding and frame clipping

Crop around each selection with context padding so text near or slightly outside its boundary can still be detected and recognized. Clip crop bounds to the frame edges.

The default context padding is 10 pixels on each side, clipped to the frame bounds. This is an initial policy, not a guarantee that every candidate accepted by the area-based overflow rule will be visible to OCR. Validate it with subtitles near selection edges. Keep the policy centralized; do not add a GUI setting unless testing shows that user-configurable padding is needed.

Padding cannot guarantee preservation of every candidate accepted by the existing area-based overflow rule. The existing full-frame-coordinate filter remains authoritative after OCR.

### Bidirectional coordinate transforms

Use each area's ROI instance to:

- Crop a full frame into ROI-local image coordinates.
- Translate crop-relative `(x, y)` points back to full-frame coordinates after detection/OCR.
- Translate full-frame points into ROI coordinates when needed.

Translate polygon vertices, not only axis-aligned boxes, so the helper works with detector and OCR outputs. Translate model results back immediately after inference. Caches, selection checks, subtitle output, and debug overlays continue to use full-frame coordinates.

### Full-frame coordinates at public API boundaries

Keep the existing detection and OCR result formats; do not add a new prediction/result type solely to mark coordinate spaces. Instead, add an optional ROI argument to the existing public detector and OCR methods.

When an ROI is supplied, each method crops the input frame, runs inference on the crop, and translates all returned boxes or polygon vertices back to full-frame coordinates before returning. When no ROI is supplied, methods retain their current full-frame behavior. The return-coordinate contract is therefore invariant: downstream code receives coordinates in the original video-frame space.

Crop-relative predictions are internal to the inference methods and must not escape them. Main extraction logic, selection filtering, caches, queues, logs, subtitle output, and debug rendering only receive full-frame-coordinate results. Do not use flags that require downstream consumers to remember whether a result was cropped, and do not apply a second translation after the method returns.

### Share ROI geometry and per-frame crops

Detection and recognition for a frame/area use crops derived from the same ROI geometry. Do not independently recalculate bounds in detector and recognizer. ROI cropping is a NumPy view and does not copy frame pixels.

In accurate mode, the candidate-frame detector and OCR used to compare subtitle text must follow the same ROI path as the later asynchronous OCR worker. Where results are cached or passed through task queues, only store/pass full-frame-coordinate results. This avoids ambiguity and double translation.

The ROI stores geometry, not image pixels. Each frame gets its own cropped image.

The minimal architecture is:

```text
original frame + optional ROI
          |
          v
existing detector/OCR public method
  - crop internally when ROI exists
  - run model on crop
  - translate predictions back before return
          |
          v
existing full-frame result format
          |
          v
main pipeline, caches, queues, filters, and output
```

If detection and OCR run on the same frame, the caller may create and reuse that frame's crop to avoid slicing twice. This must preserve the same public guarantee: results returned to the main pipeline use full-frame coordinates. Prefer a small shared internal crop-and-translate helper over duplicated translation logic.

### Preserve final selection behavior

After translating boxes to full-frame coordinates, retain the existing selection-overlap, allowed-deviation, and score checks. Cropping is a performance optimization, not a change to which recognized results are accepted.

## Implementation strategy

### Phase 1: Single-area ROI integration

The currently supported selected area is integrated end-to-end first:

1. Initialize the selected area's ROI once the extractor knows the video's frame shape. (Implemented.)
2. Add optional ROI parameters to the existing detector and OCR public methods. Preserve their current no-ROI behavior and result formats.
3. In accurate mode, pass the ROI to text detection and OCR used for subtitle-text comparison. Translate detector polygons and OCR boxes to full-frame coordinates inside those methods before returning; existing containment checks and caches then remain in full-frame coordinates.
4. Pass the area's ROI to the asynchronous OCR worker. For uncached recognition, pass the ROI into the OCR method. Queue and cache payloads remain full-frame coordinates.
5. When a custom ROI is present, let it supersede the coarse upper/lower-half crop. Without a custom ROI, preserve the existing preprocessing behavior.
6. Preserve unchanged behavior when no custom selection exists.

Steps 2–5 are implemented. Validation and real-video comparison remain.

### Phase 2: Multiple selected areas

When task plumbing is changed to process every selected area:

1. Initialize one independent ROI for every selected `SubtitleArea`.
2. Associate every crop, cache entry, and OCR result with its source area.
3. Apply the existing acceptance checks against the matching area.
4. Deduplicate overlapping detections/subtitle results when the same text is found through multiple ROI crops.

Do not combine disjoint selections into one large bounding rectangle; that would include unselected gaps and reduce the optimization.

## Validation

### Unit-level checks

- Crop bounds and image pixels for ordinary selections.
- Padding clipped at all four frame edges.
- Invalid, empty, or non-intersecting selections fail explicitly.
- Crop-relative points round-trip through `to_original()` and `to_relative()`.
- Polygon vertices and box corners translate consistently.
- Existing public detector/OCR methods return full-frame coordinates with and without an ROI.
- Crop-relative predictions never escape the inference-method boundary.
- Results are translated exactly once before cache or queue handoff.
- Mismatched frame dimensions raise an explicit error.
- A custom ROI takes precedence over upper/lower-half preprocessing; no-ROI preprocessing remains unchanged.
- Multiple areas own distinct ROIs with correct bounds.

### Pipeline checks

- Fresh OCR and cached OCR follow the same full-frame coordinate contract and area-specific ROI.
- Accurate-mode detection, comparison OCR, and asynchronous OCR all receive the same ROI.
- Boxes inside a selection remain accepted.
- Text outside an area's padded crop is not sent to detection/recognition for that ROI.
- Text crossing selection edges remains discoverable when covered by padding and is still judged by the existing overflow filter.
- No-selection extraction remains unchanged.
- Multi-area results remain associated with their source area; overlapping results are deduplicated.
- Compare a representative video before/after for subtitle text, timing, discarded out-of-selection results, and processing time.

## Non-goals

- Changing Hebrew OCR confidence calculation or the OCR model.
- Changing subtitle timing, text post-processing, or the selection acceptance formula.
- Enabling multi-area extraction as part of Phase 1.
- Exposing padding as a user-facing option before testing establishes that it is needed.
