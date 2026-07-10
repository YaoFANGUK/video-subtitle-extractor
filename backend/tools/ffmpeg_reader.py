"""
FFmpeg-based video reader — drop-in replacement for cv2.VideoCapture.

Uses system ffmpeg (full codec support including AV1/HEVC/VP9)
instead of OpenCV's bundled ffmpeg backend.

Usage:
    from backend.tools.ffmpeg_reader import FFmpegReader
    cap = FFmpegReader(video_path)
    prop = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # same API
    ret, frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    cap.release()
"""

import json
import subprocess

import cv2
import numpy as np


class FFmpegReader:
    """Sequential frame reader via ffmpeg pipe. Forward-only seeking."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self._proc = None
        self._current_frame_no = 0
        self._frame_count = 0
        self._fps = 0.0
        self._frame_width = 0
        self._frame_height = 0
        self._seek_frame = None
        self._codec_name = None
        self._opened = self._read_metadata()

    # ── metadata via ffprobe ───────────────────────────────────────

    def _read_metadata(self) -> bool:
        """Fill basic video properties from ffprobe."""
        try:
            args = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                self.video_path,
            ]
            result = subprocess.run(args, capture_output=True, text=True)
            if result.returncode != 0:
                return False
            info = json.loads(result.stdout)
            for stream in info.get("streams", []):
                if stream["codec_type"] != "video":
                    continue
                self._frame_width = stream.get("width", 0)
                self._frame_height = stream.get("height", 0)
                self._codec_name = stream.get("codec_name", "")
                num, den = (stream.get("r_frame_rate", "30/1").split("/"))
                self._fps = float(num) / float(den)
                nb_frames = stream.get("nb_frames")
                if nb_frames:
                    self._frame_count = int(nb_frames)
                break
            if self._frame_count == 0:
                duration = float(info.get("format", {}).get("duration", 0))
                self._frame_count = int(duration * self._fps) if duration else 0
            return self._frame_count > 0 and self._frame_width > 0
        except Exception:
            return False

    # ── cv2.VideoCapture compatible API ────────────────────────────

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop: int) -> float:
        """Emulate cv2.VideoCapture.get()."""
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._frame_count)
        elif prop == cv2.CAP_PROP_FPS:
            return self._fps
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._frame_height)
        elif prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._frame_width)
        elif prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self._current_frame_no)
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        """Schedule a seek. Must be forward-only (monotonic)."""
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._seek_frame = int(value)
            self._seek_ms = None
            return True
        elif prop == cv2.CAP_PROP_POS_MSEC:
            self._seek_ms = int(value)
            self._seek_frame = None
            return True
        return False

    def read(self):
        """Return (ret, frame). ret is False at EOF."""
        if self._proc is None:
            if not self._open_pipe():
                return False, None

        # If we have a seek target, advance from current position
        target = None
        if hasattr(self, "_seek_frame") and self._seek_frame is not None:
            target = self._seek_frame
            self._seek_frame = None
            self._seek_ms = None
        elif hasattr(self, "_seek_ms") and self._seek_ms is not None:
            # For MSEC seek, approximate: target_frame = ms * fps / 1000
            target = int(self._seek_ms * self._fps / 1000.0)
            self._seek_ms = None

        if target is not None and target > self._current_frame_no:
            self._skip_frames(target - self._current_frame_no)

        raw = self._proc.stdout.read(self._frame_size)
        if not raw or len(raw) < self._frame_size:
            return False, None
        self._current_frame_no += 1
        frame = (
            np.frombuffer(raw[: self._frame_size], np.uint8)
            .reshape(self._frame_height, self._frame_width, 3)
            .copy()
        )  # copy for writable array
        return True, frame

    def release(self):
        if self._proc:
            self._proc.stdout.close()
            self._proc.terminate()
            self._proc.wait(timeout=5)
            self._proc = None

    def __del__(self):
        self.release()

    # ── internal ───────────────────────────────────────────────────

    @property
    def _frame_size(self) -> int:
        return self._frame_width * self._frame_height * 3

    def _open_pipe(self) -> bool:
        try:
            self._proc = subprocess.Popen(
                self._ffmpeg_args(),
                stdout=subprocess.PIPE,
                bufsize=10**8,
            )
            return True
        except FileNotFoundError:
            print("ffmpeg not found in PATH. Install ffmpeg or use CPU mode.")
            return False

    def _ffmpeg_args(self) -> list:
        """Build ffmpeg command line flags for this video."""
        args = ["ffmpeg"]
        # Force software decoder for codecs where hardware decode commonly fails
        if self._codec_name == "av1":
            args += ["-c:v", "libdav1d"]
        args += ["-i", self.video_path]
        args += [
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vsync", "0",
            "-an",
            "-v", "quiet",
            "pipe:1",
        ]
        return args

    def _skip_frames(self, count: int):
        """Read and discard count frames from the pipe."""
        for _ in range(count):
            raw = self._proc.stdout.read(self._frame_size)
            if not raw or len(raw) < self._frame_size:
                break
        self._current_frame_no += count
