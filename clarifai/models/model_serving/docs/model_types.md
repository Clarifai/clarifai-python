Each model type requires different input and output types. The table below illustrates the relationship between supported models and their corresponding input and output types.

| Type                | Input       | Output               |
|---------------------|-------------|----------------------|
| multimodal-embedder |  image,text | EmbeddingOutput      |
| text-classifier     |  text       | ClassifierOutput     |
| text-embedder       |  text       | EmbeddingOutput      |
| text-to-image       |  text       | ImageOutput          |
| text-to-text        |  text       | TextOutput           |
| visual-classifier   |  image      | ClassifierOutput     |
| visual-detector     |  image      | VisualDetectorOutput |
| visual-embedder     |  image      | EmbeddingOutput      |
| visual-segmenter    |  image      | MasksOutput          |

Note:

* `image`: single image is RGB np.ndarray with shape of [W, H, 3]
* `text`: single text is a string in python
* `multimodal`: has more than one input types
