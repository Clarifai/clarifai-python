## Clarifai Model Prediction Output Formats.

Different models return different types of predictions and Clarifai output dataclasses aim at standardizing the output formats per model type for compatibility with the Clarifai API.

Each machine learning modality supported by the Clarifai API has a predefined dataclass output format with all attributes being of numpy ndarray type.

## Supported Formats

Usage:
```python
from clarifai.models.model_serving.models.output import VisualDetectorOutput
```
| Output Type (dataclass) | Attributes | Attribute Data Type| Attribute Shapes | Description |
| --- | --- | --- | --- | --- |
| [VisualDetectorOutput](../models/output.py) | `predicted_bboxes` | float32 | [-1, 4] | A 2D detected bounding boxes array of any length with each element array having a length of exactly 4. All bbox coordinates MUST be normalized between 0 & 1. |
| | `predicted_labels` | int32 | [-1, 1] | A 2D detected labels array of length equal to that of predicted_bboxes with each element array having a length of exactly 1.
| | `predicted_scores` | float32 | [-1, 1] | A 2D detection scores array of length equal to that of predicted_bboxes & predicted_labels with each element array having a length of exactly 1.
| | | | | |
| [ClassifierOutput](../models/output.py) | `predicted_scores` | float32 | [-1] | The softmax of the model's predictions. The index of each predicted probability as returned by the model must correspond to the label index in the labels.txt file |
| | | | | |
| [TextOutput](../models/output.py) | `predicted_text` | string | [1] | Predicted text from a model |
| | | | | |
| [EmbeddingOutput](../models/output.py) | `embedding_vector` | float32 | [-1] | The embedding vector (image or text embedding) returned by a model |
| | | | | |
| [MasksOutput](../models/output.py) | `predicted_mask` | int64 | [-1, -1] | The model predicted image mask. The predicted class indices must be assigned to the corresponding image pixels in the mask where that class is predicted by the model. |
| | | | | |
| [ImageOutput](../models/output.py) | `image` | unint8 | [-1, -1, 3] | The model predicted/generated image |
| | | | | |
