DATASET_UPLOAD_TASKS = [
    "visual_classification",
    "text_classification",
    "visual_detection",
    "visual_segmentation",
    "visual_captioning",
    "multimodal_dataset",
]

TASK_TO_ANNOTATION_TYPE = {
    "visual_classification": {"concepts": "labels"},
    "text_classification": {"concepts": "labels"},
    "visual_captioning": {"concepts": "labels"},
    "visual_detection": {"bboxes": "bboxes"},
    "visual_segmentation": {"polygons": "polygons"},
}

MAX_RETRIES = 2

CONTENT_TYPE = {"json": "application/json", "zip": "application/zip"}
