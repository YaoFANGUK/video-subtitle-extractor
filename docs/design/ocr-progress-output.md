# OCR Progress and Console Output

## Status

OCR confidence messages and OCR task-progress events are forwarded from the OCR worker process to the extraction process. The extraction process owns rendering the keyframe and OCR progress bars, avoiding competing `tqdm` redraws between processes.

## Problem

Extraction runs work in multiple processes. The parent extraction process displays keyframe progress, while the asynchronous OCR pipeline has a producer and consumer running in a child process. The consumer reports recognition results and the producer tracks OCR task progress.

Previously, both the parent and child processes wrote progress bars and recognition messages directly to stdout. `tqdm.write()` coordinates output within a process, but its output lock does not coordinate independent processes. A child-process write can therefore occur during a parent-process redraw, making confidence messages appear on the same line as a progress bar or leaving stale redraw fragments.

## Design decisions

- Keep the existing multiprocessing progress queue as the route for OCR status events.
- Do not let the OCR child process render a progress bar or write recognition-result lines directly to stdout.
- Render parent-owned terminal progress bars and terminal log messages from the extraction process.
- Preserve the existing GUI log callback behavior. In the GUI, `append_output` forwards log messages to the GUI through its existing queue.
- Keep Paddle, CUDA, and other native-library output outside this application-level routing. Such libraries may still write directly to stdout or stderr and disturb terminal redraws.

## Event flow

```text
OCR producer thread (child process)         OCR consumer thread (child process)
  -- ("bar_start", total) --------------------->|
  -- ("bar_update", frame_number) ------------>|
  -- ("bar_finish",) ------------------------->|  multiprocessing progress queue
                                                |
                                                v
                                  extraction-process progress thread
                                    - owns the OCR tqdm bar
                                    - forwards log events to append_output
                                                |
                                                v
                                  parent terminal or GUI log callback
```

The keyframe `tqdm` bar is also owned by the extraction process, so both application progress bars are rendered in that process. OCR events carried by the progress queue are:

| Event | Meaning |
| --- | --- |
| `("bar_start", total)` | Initialize the OCR task bar with its total task count. |
| `("bar_update", frame_number)` | Advance the OCR task bar to the latest queued frame number. |
| `("bar_finish",)` | Complete and close the OCR task bar. |
| `("log", message)` | Forward a confidence/result message through `append_output`. |

The existing progress tuples for OCR completion and processed-count reporting remain on the same queue. The progress thread handles these events as well, updating the GUI/application OCR progress independently of terminal bar redraws.

OCR confidence/result messages also include the sample's position in the source video, formatted as `Finished  XX.X% | Confidence: ...` (percentage right-aligned to 5 characters so `99.5%` and `100.0%` line up). The producer computes a 0..1 position ratio per task and passes it as the fifth `ocr_queue` element (the end sentinel carries `1.0`). For normal tasks it is `frame_no / frame_count`; for VideoSubFinder tasks it is `timestamp_ms / duration_ms`, because their `frame_no` is a lookup key (`int(ms / fps)`) and not a real frame number, and using it directly made results show 100% long before the end of the video. The OCR bar is advanced from the same ratio. It indicates the result's timeline position, not the fraction of queued OCR tasks completed.

## Output boundary

In the command-line path, `append_output` uses `tqdm.write()` in the extraction process so an application log message clears and redraws the active bars cleanly. In the GUI path, `append_output` is replaced with the GUI remote-call callback, which forwards the message to the GUI log view rather than writing it directly to the terminal.

The GUI runs extraction in a separate process and renders its own progress widgets and log view. Since those GUI log writes and terminal progress-bar writes belong to different processes, GUI-launched extraction disables terminal `tqdm` bars. Command-line extraction keeps the keyframe and OCR progress bars, both rendered by the extraction process.

This synchronization applies to application-owned output routed through these APIs. It does not capture arbitrary writes from Paddle or native CUDA/cuDNN libraries.

## Validation

Unit tests verify that OCR result messages use the supplied callback instead of writing through the child process's `tqdm`, and that the OCR producer sends bar events through the progress queue instead of creating its own bar. End-to-end visual behavior should also be checked with a representative extraction because third-party native output is not covered by the application queue.
