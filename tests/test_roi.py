import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from backend.bean.subtitle_area import SubtitleArea
from backend.tools.roi import SubtitleROI
from backend.tools.ocr import OcrRecogniser
from backend.tools.subtitle_detect import SubtitleDetect
from backend.tools import subtitle_ocr


def make_coordinate_image(height: int, width: int) -> np.ndarray:
    y, x = np.indices((height, width))
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = x % 256
    image[:, :, 1] = y % 256
    image[:, :, 2] = (x + y) % 256
    return image


class FakeTextDetector:
    def __init__(self, polygons: np.ndarray) -> None:
        self.polygons = polygons
        self.image = None

    def predict(self, image: np.ndarray) -> list[dict[str, np.ndarray]]:
        self.image = image
        return [{"dt_polys": self.polygons}]


class FakeOCR:
    def __init__(self, polygons: np.ndarray) -> None:
        self.polygons = polygons
        self.image = None

    def predict_iter(self, image: np.ndarray) -> list[dict[str, object]]:
        self.image = image
        return [{
            "dt_polys": self.polygons,
            "rec_texts": ["sample"],
            "rec_scores": [0.9],
        }]


class FakeFrameOCR:
    def __init__(self) -> None:
        self.image = None
        self.roi = None

    def predict(
        self,
        image: np.ndarray,
        roi: SubtitleROI | None = None,
    ) -> tuple[list[list[tuple[int, int]]], list[tuple[str, float]]]:
        self.image = image
        self.roi = roi
        return (
            [[(35, 40), (45, 40), (45, 50), (35, 50)]],
            [("subtitle", 0.99)],
        )


class UnexpectedOCRCall:
    def predict(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("cached OCR results should not run inference again")


class SubtitleROITest(unittest.TestCase):
    def test_default_padding_is_ten_pixels(self) -> None:
        frame = make_coordinate_image(height=40, width=60)
        roi = SubtitleROI(
            frame.shape,
            SubtitleArea(ymin=10, ymax=20, xmin=15, xmax=30),
        )

        self.assertEqual(roi.bounds, (5, 0, 40, 30))
        self.assertEqual(roi.crop(frame).shape, (30, 35, 3))

    def test_crop_returns_padded_coordinate_encoded_region(self) -> None:
        frame = make_coordinate_image(height=40, width=60)
        roi = SubtitleROI(
            frame.shape,
            SubtitleArea(ymin=10, ymax=20, xmin=15, xmax=30),
            padding=3,
        )

        cropped = roi.crop(frame)

        self.assertEqual(roi.bounds, (12, 7, 33, 23))
        self.assertEqual(cropped.shape, (16, 21, 3))
        self.assertEqual(cropped[0, 0].tolist(), [12, 7, 19])
        self.assertEqual(cropped[-1, -1].tolist(), [32, 22, 54])
        np.testing.assert_array_equal(cropped, frame[7:23, 12:33])

    def test_crop_clips_padding_at_frame_edges(self) -> None:
        frame = make_coordinate_image(height=40, width=60)
        roi = SubtitleROI(
            frame.shape,
            SubtitleArea(ymin=0, ymax=10, xmin=0, xmax=12),
            padding=5,
        )

        cropped = roi.crop(frame)

        self.assertEqual(roi.bounds, (0, 0, 17, 15))
        self.assertEqual(cropped.shape, (15, 17, 3))
        np.testing.assert_array_equal(cropped, frame[:15, :17])

    def test_coordinate_transforms_round_trip(self) -> None:
        frame = make_coordinate_image(height=40, width=60)
        roi = SubtitleROI(
            frame.shape,
            SubtitleArea(ymin=10, ymax=20, xmin=15, xmax=30),
            padding=3,
        )
        original_points = [(12, 7), (20, 15), (32, 22)]

        relative_points = roi.to_relative(original_points)

        self.assertEqual(relative_points, [(0, 0), (8, 8), (20, 15)])
        self.assertEqual(roi.to_original(relative_points), original_points)

    def test_subtitle_area_initializes_and_owns_roi(self) -> None:
        area = SubtitleArea(ymin=10, ymax=20, xmin=15, xmax=30)

        self.assertIsNone(area.roi)
        roi = area.initialize_roi((40, 60))

        self.assertIs(area.roi, roi)
        self.assertEqual(area.roi.bounds, (5, 0, 40, 30))

    def test_subtitle_area_roi_survives_process_serialization(self) -> None:
        area = SubtitleArea(ymin=30, ymax=60, xmin=30, xmax=60)
        area.initialize_roi((100, 120))

        restored_area = pickle.loads(pickle.dumps(area))

        self.assertEqual(restored_area.roi.bounds, area.roi.bounds)

    def test_detector_returns_full_frame_polygons_when_roi_is_used(self) -> None:
        frame = make_coordinate_image(height=100, width=120)
        roi = SubtitleROI(
            frame.shape,
            SubtitleArea(ymin=30, ymax=60, xmin=30, xmax=60),
            padding=10,
        )
        local_polygons = np.array(
            [[[2, 10], [8, 10], [8, 14], [2, 14]]],
            dtype=np.float32,
        )
        fake_detector = FakeTextDetector(local_polygons)
        detector = object.__new__(SubtitleDetect)
        detector.text_detector = fake_detector

        polygons, _ = detector.detect_subtitle(frame, roi=roi)

        self.assertEqual(fake_detector.image.shape, (50, 50, 3))
        np.testing.assert_array_equal(
            polygons,
            np.array([[[22, 30], [28, 30], [28, 34], [22, 34]]], dtype=np.float32),
        )

    def test_detector_without_roi_keeps_full_frame_input_and_coordinates(self) -> None:
        frame = make_coordinate_image(height=100, width=120)
        local_polygons = np.array(
            [[[2, 10], [8, 10], [8, 14], [2, 14]]],
            dtype=np.float32,
        )
        fake_detector = FakeTextDetector(local_polygons)
        detector = object.__new__(SubtitleDetect)
        detector.text_detector = fake_detector

        polygons, _ = detector.detect_subtitle(frame)

        self.assertIs(fake_detector.image, frame)
        np.testing.assert_array_equal(polygons, local_polygons)

    def test_ocr_returns_full_frame_boxes_when_roi_is_used(self) -> None:
        frame = make_coordinate_image(height=100, width=120)
        roi = SubtitleROI(
            frame.shape,
            SubtitleArea(ymin=30, ymax=60, xmin=30, xmax=60),
            padding=10,
        )
        local_polygons = np.array(
            [[[2, 10], [8, 10], [8, 14], [2, 14]]],
            dtype=np.float32,
        )
        fake_ocr = FakeOCR(local_polygons)
        recogniser = object.__new__(OcrRecogniser)
        recogniser.recogniser = fake_ocr

        boxes, results = recogniser.predict(frame, roi=roi)

        self.assertEqual(fake_ocr.image.shape, (50, 50, 3))
        self.assertEqual(boxes, [[(22, 30), (28, 30), (28, 34), (22, 34)]])
        self.assertEqual(results, [("sample", 0.9)])

    def test_ocr_without_roi_keeps_full_frame_input_and_coordinates(self) -> None:
        frame = make_coordinate_image(height=100, width=120)
        local_polygons = np.array(
            [[[2, 10], [8, 10], [8, 14], [2, 14]]],
            dtype=np.float32,
        )
        fake_ocr = FakeOCR(local_polygons)
        recogniser = object.__new__(OcrRecogniser)
        recogniser.recogniser = fake_ocr

        boxes, results = recogniser.predict(frame)

        self.assertIs(fake_ocr.image, frame)
        self.assertEqual(boxes, [[(2, 10), (8, 10), (8, 14), (2, 14)]])
        self.assertEqual(results, [("sample", 0.9)])

    def test_ocr_uses_initialized_accelerator_for_model_device(self) -> None:
        accelerator = SimpleNamespace(has_cuda=lambda: True)
        model_config = SimpleNamespace(
            DET_MODEL_PATH="det",
            REC_MODEL_PATH="rec",
            DET_MODEL_NAME=None,
            REC_MODEL_NAME=None,
        )

        with (
            patch(
                "backend.tools.ocr.HardwareAccelerator.instance",
                return_value=accelerator,
            ),
            patch(
                "backend.tools.ocr.PaddleModelConfig",
                return_value=model_config,
            ) as model_config_factory,
            patch("backend.tools.ocr.PaddleOCR") as paddle_ocr,
        ):
            recogniser = OcrRecogniser()
            recogniser.init_model()

        self.assertIs(recogniser.hardware_accelerator, accelerator)
        model_config_factory.assert_called_once_with(accelerator)
        self.assertEqual(paddle_ocr.call_args.kwargs["device"], "gpu:0")

    def test_subtitle_extraction_passes_roi_to_ocr_and_keeps_frame_coordinates(self) -> None:
        frame = make_coordinate_image(height=100, width=120)
        area = SubtitleArea(ymin=30, ymax=60, xmin=30, xmax=60)
        roi = area.initialize_roi(frame.shape)
        fake_ocr = FakeFrameOCR()
        raw_subtitles: list[str] = []
        options = SimpleNamespace(
            REC_CHAR_TYPE="he",
            DROP_SCORE=0.5,
            SUB_AREA_DEVIATION_RATE=0,
            DEBUG_OCR_LOSS=False,
        )

        with patch.object(subtitle_ocr.tqdm, "write"):
            subtitle_ocr.extract_subtitles(
                {"i": 1},
                fake_ocr,
                frame,
                raw_subtitles,
                area,
                options,
                None,
                None,
                "",
                roi=roi,
            )

        self.assertIs(fake_ocr.image, frame)
        self.assertIs(fake_ocr.roi, roi)
        self.assertEqual(
            raw_subtitles,
            ["00000001\t(35, 45, 40, 50)\tsubtitle\n"],
        )

    def test_subtitle_extraction_sends_ocr_log_to_callback(self) -> None:
        frame = make_coordinate_image(height=100, width=120)
        area = SubtitleArea(ymin=30, ymax=60, xmin=30, xmax=60)
        fake_ocr = FakeFrameOCR()
        raw_subtitles: list[str] = []
        log_messages: list[str] = []
        options = SimpleNamespace(
            REC_CHAR_TYPE="he",
            DROP_SCORE=0.5,
            SUB_AREA_DEVIATION_RATE=0,
            DEBUG_OCR_LOSS=False,
        )

        with patch.object(subtitle_ocr.tqdm, "write") as tqdm_write:
            subtitle_ocr.extract_subtitles(
                {"i": 1},
                fake_ocr,
                frame,
                raw_subtitles,
                area,
                options,
                None,
                None,
                "",
                log_callback=log_messages.append,
            )

        self.assertEqual(len(log_messages), 1)
        self.assertIn("subtitle", log_messages[0])
        tqdm_write.assert_not_called()

    def test_ocr_progress_message_shows_source_video_position(self) -> None:
        message = subtitle_ocr.format_ocr_progress_message(
            0.375,
            "√ Confidence: 97.5% Result: sample",
        )

        self.assertEqual(
            message,
            "Finished  37.5% | √ Confidence: 97.5% Result: sample",
        )

    def test_ocr_progress_message_aligns_percentages(self) -> None:
        self.assertEqual(
            subtitle_ocr.format_ocr_progress_message(0.995, "x")[:16],
            "Finished  99.5% ",
        )
        self.assertEqual(
            subtitle_ocr.format_ocr_progress_message(1.0, "x")[:16],
            "Finished 100.0% ",
        )

    def test_vsf_task_progress_uses_timestamp_not_frame_key(self) -> None:
        task_queue = Mock()
        # VSF tasks key frames as int(ms / fps); 90% into a 100 s, 25 fps video.
        total_frames = 2500
        total_ms = 90000
        task_queue.get.side_effect = [
            (total_frames, int(total_ms / 25), None, None, total_ms, None),
            (total_frames, -1, None, None, None, None),
        ]
        ocr_queue = Mock()
        progress_queue = Mock()
        capture = Mock()
        capture.get.return_value = 25.0
        capture.read.return_value = (True, np.zeros((4, 4, 3), dtype=np.uint8))

        with patch.object(subtitle_ocr.cv2, "VideoCapture", return_value=capture):
            subtitle_ocr.ocr_task_producer(ocr_queue, task_queue, progress_queue, "v.mp4", "raw.txt")

        self.assertAlmostEqual(ocr_queue.put.call_args_list[0].args[0][4], 0.9)
        self.assertIn(("bar_update", 2250), [c.args[0] for c in progress_queue.put.call_args_list])

    def test_ocr_producer_sends_progress_events_instead_of_drawing_its_own_bar(self) -> None:
        task_queue = Mock()
        task_queue.get.side_effect = [
            (20, 10, None, None, None, None),
            (20, -1, None, None, None, None),
        ]
        ocr_queue = Mock()
        progress_queue = Mock()
        capture = Mock()
        capture.read.return_value = (True, np.zeros((4, 4, 3), dtype=np.uint8))

        with patch.object(subtitle_ocr.cv2, "VideoCapture", return_value=capture):
            subtitle_ocr.ocr_task_producer(
                ocr_queue,
                task_queue,
                progress_queue,
                "video.mp4",
                "raw.txt",
            )

        self.assertEqual(
            [call.args[0] for call in progress_queue.put.call_args_list],
            [
                ("bar_start", 20),
                ("bar_update", 10),
                (-2, 1),
                ("bar_finish",),
            ],
        )
        self.assertEqual(
            ocr_queue.put.call_args_list,
            [
                unittest.mock.call((
                    10,
                    capture.read.return_value[1],
                    None,
                    None,
                    0.5,
                )),
                unittest.mock.call((-1, 1, None, None, 1.0)),
            ],
        )
        capture.release.assert_called_once()

    def test_cached_full_frame_boxes_are_not_translated_again(self) -> None:
        frame = make_coordinate_image(height=100, width=120)
        area = SubtitleArea(ymin=30, ymax=60, xmin=30, xmax=60)
        roi = area.initialize_roi(frame.shape)
        raw_subtitles: list[str] = []
        options = SimpleNamespace(
            REC_CHAR_TYPE="he",
            DROP_SCORE=0.5,
            SUB_AREA_DEVIATION_RATE=0,
            DEBUG_OCR_LOSS=False,
        )
        full_frame_boxes = [[(35, 40), (45, 40), (45, 50), (35, 50)]]

        with patch.object(subtitle_ocr.tqdm, "write"):
            subtitle_ocr.extract_subtitles(
                {"i": 1},
                UnexpectedOCRCall(),
                frame,
                raw_subtitles,
                area,
                options,
                full_frame_boxes,
                [("cached", 0.99)],
                "",
                roi=roi,
            )

        self.assertEqual(
            raw_subtitles,
            ["00000001\t(35, 45, 40, 50)\tcached\n"],
        )


if __name__ == "__main__":
    unittest.main()
