import os
import tempfile
from io import BytesIO
from typing import Dict, Iterator, List

import cv2
import torch
from PIL import Image as PILImage

from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_types import Concept, Frame, Image, Region
from clarifai.utils.logging import logger


class VisualDetectorClass(ModelClass):
    """Base class for visual detection models supporting image and video processing."""

    @staticmethod
    def preprocess_image(image_bytes: bytes) -> PILImage:
        """Convert image bytes to PIL Image."""
        return PILImage.open(BytesIO(image_bytes)).convert("RGB")

    @staticmethod
    def video_to_frames(video_bytes: bytes) -> Iterator[Frame]:
        """Convert video bytes to frames.

        Args:
            video_bytes: Raw video data in bytes

        Yields:
            Frame with JPEG encoded frame data as bytes and timestamp in milliseconds
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            temp_video_file.write(video_bytes)
            temp_video_path = temp_video_file.name
            logger.debug(f"temp_video_path: {temp_video_path}")

            video = cv2.VideoCapture(temp_video_path)
            logger.debug(f"video opened: {video.isOpened()}")

            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                # Get frame timestamp in milliseconds
                timestamp_ms = video.get(cv2.CAP_PROP_POS_MSEC)
                frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
                yield Frame(image=Image(bytes=frame_bytes), time=timestamp_ms)

            video.release()
            os.unlink(temp_video_path)

    @staticmethod
    def process_detections(
        results: List[Dict[str, torch.Tensor]], threshold: float, model_labels: Dict[int, str]
    ) -> List[List[Region]]:
        """Convert model outputs into a structured format of detections.

        Args:
            results: Raw detection results from model
            threshold: Confidence threshold for detections
            model_labels: Dictionary mapping label indices to names

        Returns:
            List of lists containing Region objects for each detection
        """
        outputs = []
        for result in results:
            detections = []
            for score, label_idx, box in zip(result["scores"], result["labels"], result["boxes"]):
                if score > threshold:
                    label = model_labels[label_idx.item()]
                    detections.append(
                        Region(
                            box=box.tolist(), concepts=[Concept(name=label, value=score.item())]
                        )
                    )
            outputs.append(detections)
        return outputs
