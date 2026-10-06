from collections.abc import Sequence
from math import ceil, floor
from typing import Any

from backend.bean.subtitle_area import SubtitleArea


DEFAULT_ROI_PADDING = 10


class SubtitleROI:
    """A reusable padded crop and coordinate transform for one video's frames."""

    def __init__(
        self,
        frame_shape: Sequence[int],
        selection: SubtitleArea,
        padding: int = DEFAULT_ROI_PADDING,
    ) -> None:
        if len(frame_shape) < 2:
            raise ValueError("frame_shape must include height and width")
        if padding < 0:
            raise ValueError("padding must be non-negative")

        self.frame_height = int(frame_shape[0])
        self.frame_width = int(frame_shape[1])
        if self.frame_height <= 0 or self.frame_width <= 0:
            raise ValueError("frame dimensions must be positive")

        left = floor(min(selection.xmin, selection.xmax)) - padding
        top = floor(min(selection.ymin, selection.ymax)) - padding
        right = ceil(max(selection.xmin, selection.xmax)) + padding
        bottom = ceil(max(selection.ymin, selection.ymax)) + padding

        self.left = max(0, left)
        self.top = max(0, top)
        self.right = min(self.frame_width, right)
        self.bottom = min(self.frame_height, bottom)
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("selection does not intersect the frame")

        self.origin = (self.left, self.top)
        self._to_original_offset = self.origin
        self._to_relative_offset = (-self.left, -self.top)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Return the crop bounds as (left, top, right, bottom)."""
        return self.left, self.top, self.right, self.bottom

    def crop(self, image: Any) -> Any:
        """Crop a full video frame using this ROI's precomputed bounds."""
        if image.shape[0] != self.frame_height or image.shape[1] != self.frame_width:
            raise ValueError(
                "image dimensions do not match the frame dimensions used to create the ROI"
            )
        return image[self.top:self.bottom, self.left:self.right]

    def to_original(
        self,
        points: Sequence[Sequence[int | float]],
    ) -> list[tuple[int | float, int | float]]:
        """Translate crop-relative (x, y) points to full-frame coordinates."""
        return self._translate(points, self._to_original_offset)

    def to_relative(
        self,
        points: Sequence[Sequence[int | float]],
    ) -> list[tuple[int | float, int | float]]:
        """Translate full-frame (x, y) points to crop-relative coordinates."""
        return self._translate(points, self._to_relative_offset)

    @staticmethod
    def _translate(
        points: Sequence[Sequence[int | float]],
        offset: tuple[int, int],
    ) -> list[tuple[int | float, int | float]]:
        dx, dy = offset
        translated = []
        for point in points:
            if len(point) != 2:
                raise ValueError("each coordinate must contain exactly x and y")
            translated.append((point[0] + dx, point[1] + dy))
        return translated
