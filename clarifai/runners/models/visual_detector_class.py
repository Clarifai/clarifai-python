import os
import tempfile
from io import BytesIO
from typing import Dict, Iterator, List, Optional

import cv2
import torch
import yaml
from PIL import Image as PILImage

from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_types import Concept, Frame, Image, Region
from clarifai.utils.logging import logger


class VisualDetectorClass(ModelClass):
    """Base class for visual detection models supporting image and video processing."""

    @staticmethod
    def load_concepts_from_config(config_path: str) -> Optional[Dict[int, Dict[str, str]]]:
        """Load concepts from config.yaml and return a mapping of index to concept info.

        Args:
            config_path: Path to the config.yaml file or the model directory containing it.

        Returns:
            Dictionary mapping label indices to concept info {'id': id, 'name': name},
            or None if no concepts are found in the config.

        Example:
            >>> concepts_map = VisualDetectorClass.load_concepts_from_config('/path/to/model')
            >>> # Returns: {0: {'id': '0', 'name': 'person'}, 1: {'id': '1', 'name': 'bicycle'}, ...}
        """
        # Handle both file path and directory path
        if os.path.isdir(config_path):
            config_file = os.path.join(config_path, 'config.yaml')
        else:
            config_file = config_path

        if not os.path.exists(config_file):
            logger.warning(f"Config file not found at {config_file}")
            return None

        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            concepts = config.get('concepts')
            if not concepts:
                return None

            # Build mapping from index to concept info
            concepts_map = {}
            for idx, concept in enumerate(concepts):
                concept_id = concept.get('id', str(idx))
                concept_name = concept.get('name', concept_id)
                concepts_map[idx] = {'id': concept_id, 'name': concept_name}

            logger.info(f"Loaded {len(concepts_map)} concepts from config")
            return concepts_map
        except Exception as e:
            logger.warning(f"Failed to load concepts from config: {e}")
            return None

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
        results: List[Dict[str, torch.Tensor]],
        threshold: float,
        model_labels: Dict[int, str],
        concepts_map: Optional[Dict[int, Dict[str, str]]] = None,
    ) -> List[List[Region]]:
        """Convert model outputs into a structured format of detections.

        Args:
            results: Raw detection results from model
            threshold: Confidence threshold for detections
            model_labels: Dictionary mapping label indices to names (from model config)
            concepts_map: Optional dictionary mapping label indices to concept info
                {'id': id, 'name': name} from config.yaml. When provided, concept IDs
                will be taken from this map. Use load_concepts_from_config() to create this.

        Returns:
            List of lists containing Region objects for each detection
        """
        outputs = []
        for result in results:
            detections = []
            for score, label_idx, box in zip(result["scores"], result["labels"], result["boxes"]):
                if score > threshold:
                    idx = label_idx.item()
                    # Use concepts_map if available, otherwise fall back to model_labels
                    if concepts_map and idx in concepts_map:
                        concept_info = concepts_map[idx]
                        concept = Concept(
                            id=concept_info['id'],
                            name=concept_info['name'],
                            value=score.item(),
                        )
                    else:
                        label = model_labels[idx]
                        concept = Concept(name=label, value=score.item())
                    detections.append(Region(box=box.tolist(), concepts=[concept]))
            outputs.append(detections)
        return outputs
