DATASET_UPLOAD_TASKS = [
    "visual_classification", "text_classification", "visual_detection", "visual_segmentation",
    "visual_captioning"
]

TASK_TO_ANNOTATION_TYPE = {
    "visual_classification": {
        "concepts": "labels"
    },
    "text_classification": {
        "concepts": "labels"
    },
    "visual_captioning": {
        "concepts": "labels"
    },
    "visual_detection": {
        "bboxes": "bboxes"
    },
    "visual_segmentation": {
        "polygons": "polygons"
    },
}
