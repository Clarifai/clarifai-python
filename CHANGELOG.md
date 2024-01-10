## [[10.0.0]](https://github.com/Clarifai/clarifai-python/releases/tag/10.0.0) - [PyPI](https://pypi.org/project/clarifai/10.0.0/) - 2024-01-10

### Fixed
- Changed demo link in README (#260)
- Fixed Mulitmodal input bug (#261)

### Changed
- Workflow predict retry time to 10 minutes (#266)
- Update clarifai-grpc to 10.0.1 (#267)

### Added
- Test Cases for Model Upload (#256)
- Download Inputs functionality (#263)
- Added RAG base class (#262)
- RAG Chat Method (#264)
- RAG Upload Method (#265)

### Removed
- Model upload examples moved to examples repo (#258)

## [[9.11.1]](https://github.com/Clarifai/clarifai-python/releases/tag/9.11.1) - [PyPI](https://pypi.org/project/clarifai/9.11.1/) - 2023-12-29

### Fixed
- Use specific URL method for apps (#257)

### Changed
- Loosen requirement constraints (#243)
- Update clarifai-grpc to 9.11.5

### Added
- Support Rank for PostInputsSearch (#255)


## [[9.11.0]](https://github.com/Clarifai/clarifai-python/releases/tag/9.11.0) - [PyPI](https://pypi.org/project/clarifai/9.11.0/) - 2023-12-11

### Fixed
- CocoDetectionDataloader bug (#241)

### Changed
- Codeql Change (#241)
- Seperate tests requiring secrets (#233)
- SDK Pending tasks (#232)
    - add retry for workflow predict
    - add constants for max inputs count in predict
    - change annotation proto to bbox
    - add search to README.md
    - Add CHANGELOG.md
- Updated runner logic with parallel and error catching (#238)
- Removing internal_only Training Params (#231)
- Remove pytest requirement (#225)
- Remove omegaconf requirement (#235)
- Update clarifai-grpc to 9.11.0

### Added
- Support multimodal inputs for inference (#239)
- Ensure support for Python 3.10-3.12 (#226)

## [[9.10.4]](https://github.com/Clarifai/clarifai-python/releases/tag/9.10.4) - [PyPI](https://pypi.org/project/clarifai/9.10.4/) - 2023-11-23

### Fixed
- Add MANIFEST.in back to include .css files

## [[9.10.3]](https://github.com/Clarifai/clarifai-python/releases/tag/9.10.3) - [PyPI](https://pypi.org/project/clarifai/9.10.3/) - 2023-11-23

### Added
- Support Dataset Upload Status
- Support PAT as arg

### Removed
- SDK cleanup(docs, examples, symlink to clarifai_utils, clarifai.auth)

### Changed
 - Refactor dataset upload process(loaders, dataloader)

## [[9.10.2]](https://github.com/Clarifai/clarifai-python/releases/tag/9.10.2) - [PyPI](https://pypi.org/project/clarifai/9.10.2/) - 2023-11-17

### Fixed
- Fix Search top_k bug

## [[9.10.1]](https://github.com/Clarifai/clarifai-python/releases/tag/9.10.1) - [PyPI](https://pypi.org/project/clarifai/9.10.1/) - 2023-11-16

### Added
- Model Training in SDK.
- Tests for Model Training.

### Fixed
- Fix base_url bug in passing while chained

### Changed
- Moving Pycocotools requirement to extras(clarifai[all]).

## [[9.10.0]](https://github.com/Clarifai/clarifai-python/releases/tag/9.10.0) - [PyPI](https://pypi.org/project/clarifai/9.10.0/) - 2023-11-06

### Added
- Support for model inference params
- PostInputsSearch Support

### Changed
- Bump clarifai_grpc==9.10.0

## [[9.9.3]](https://github.com/Clarifai/clarifai-python/releases/tag/9.9.3) - [PyPI](https://pypi.org/project/clarifai/9.9.3/) - 2023-10-16

### Added
- Pagination in listing
- Support list_annotations
- Supports custom metadata in dataloader, upload_from_csv

### Changed
- Set clarifai_grpc to 9.8.1
- Reuse requirements.txt in setup.py

## [[9.9.2]](https://github.com/Clarifai/clarifai-python/releases/tag/9.9.2) - [PyPI](https://pypi.org/project/clarifai/9.9.2/) - 2023-10-11

### Added
- Support Annotation Download

### Fixed
- Fix critical Version file not found bug in 9.9.1

## [[9.9.1]](https://github.com/Clarifai/clarifai-python/releases/tag/9.9.1) - [PyPI](https://pypi.org/project/clarifai/9.9.1/) - 2023-10-10

### Changed
- Reuse Version number from Version file

## [[9.9.0]](https://github.com/Clarifai/clarifai-python/releases/tag/9.9.0) - [PyPI](https://pypi.org/project/clarifai/9.9.0/) - 2023-10-06

### Added
- Support Vector Search

### Fixed
- Workflow Create Bugs

## [[9.8.2]](https://github.com/Clarifai/clarifai-python/releases/tag/9.8.2) - [PyPI](https://pypi.org/project/clarifai/9.8.2/) - 2023-09-26

### Added
- Support Workflow Create, Export

## [[9.8.1]](https://github.com/Clarifai/clarifai-python/releases/tag/9.8.1) - [PyPI](https://pypi.org/project/clarifai/9.8.1/) - 2023-09-12

### Changed
- Bump clarifai_grpc to 9.8.1

## [[9.8.0]](https://github.com/Clarifai/clarifai-python/releases/tag/9.8.0) - [PyPI](https://pypi.org/project/clarifai/9.8.0/) - 2023-09-06

### Changed
- Bump clarifai_grpc to 9.8.0

## [[9.7.6]](https://github.com/Clarifai/clarifai-python/releases/tag/9.7.6) - [PyPI](https://pypi.org/project/clarifai/9.7.6/) - 2023-08-27

### Changed
- Bump clarifai_grpc to 9.7.4

## [[9.7.5]](https://github.com/Clarifai/clarifai-python/releases/tag/9.7.5) - [PyPI](https://pypi.org/project/clarifai/9.7.5/) - 2023-08-25

### Added
- Model Serving Support

## [[9.7.4]](https://github.com/Clarifai/clarifai-python/releases/tag/9.7.4) - [PyPI](https://pypi.org/project/clarifai/9.7.4/) - 2023-08-25

### Fixed
- PyPi build issues

< [[9.7.3] Clarifai Python Utils](https://github.com/Clarifai/clarifai-python-utils)
