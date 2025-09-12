import os
import tempfile
from io import BytesIO
from typing import Dict, Iterator, List

import cv2
import torch
from PIL import Image as PILImage

from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_types import Concept, Frame, Image
from clarifai.utils.logging import logger


class VisualClassifierClass(ModelClass):
    """Base class for visual classification models supporting image and video processing."""

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
    def process_concepts(
        logits: torch.Tensor, model_labels: Dict[int, str]
    ) -> List[List[Concept]]:
        """Convert model logits into a structured format of concepts.

        Args:
            logits: Model output logits as a tensor (batch_size x num_classes)
            model_labels: Dictionary mapping label indices to label names

        Returns:
            List of lists containing Concept objects for each input in the batch
        """
        outputs = []
        for logit in logits:
            probs = torch.softmax(logit, dim=-1)
            sorted_indices = torch.argsort(probs, dim=-1, descending=True)
            output_concepts = []
            for idx in sorted_indices:
                concept = Concept(name=model_labels[idx.item()], value=probs[idx].item())
                output_concepts.append(concept)
            outputs.append(output_concepts)
        return outputs
