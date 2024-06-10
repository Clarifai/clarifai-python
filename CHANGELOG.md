## [[10.5.0]](https://github.com/Clarifai/clarifai-python/releases/tag/10.5.0) - [PyPI](https://pypi.org/project/clarifai/10.5.0/) - 2024-06-10

### Changed
- Upgrade to clarifai-grpc 10.5.0 [(#345)](https://github.com/Clarifai/clarifai-python/pull/343)


## [[10.3.3]](https://github.com/Clarifai/clarifai-python/releases/tag/10.3.3) - [PyPI](https://pypi.org/project/clarifai/10.3.3/) - 2024-05-07

### Changed
- Upgrade to clarifai-grpc 10.3.4 [(#343)](https://github.com/Clarifai/clarifai-python/pull/343)
- RAG apps, workflows and other resources automatically setup now use UUIDs in their IDs instead of timestamps to avoid races. [(#343)](https://github.com/Clarifai/clarifai-python/pull/343)

### Fixed
- Fixed issue with `get_upload_status` overriding `log_warnings` table in log file. [(#342)](https://github.com/Clarifai/clarifai-python/pull/342)
- Use UUIDs in tests to avoid race conditions with timestamps. [(#343)](https://github.com/Clarifai/clarifai-python/pull/343)
- Hardcoded shcema package to 0.7.5 as it introduced breaking changes. [(#343)](https://github.com/Clarifai/clarifai-python/pull/343)


## [[10.3.2]](https://github.com/Clarifai/clarifai-python/releases/tag/10.3.2) - [PyPI](https://pypi.org/project/clarifai/10.3.2/) - 2024-05-03

### Added
- Flag to download model. If *export_dir* param in  ```Model().export()``` is provided, the exported model will be saved in the specified directory else export status will be shown.[(#337)](https://github.com/Clarifai/clarifai-python/pull/337)
- Label ID support in Dataloaders(*label_ids* param) and get_proto functions in Inputs class.[(#338)](https://github.com/Clarifai/clarifai-python/pull/338)

### Changed
- Logger for ```Inputs().upload_annotations``` to show full details of failed annotations.[(#339)](https://github.com/Clarifai/clarifai-python/pull/339)

### Fixed
- RAG upload bug by changing llama-index-core version to 0.10.24 in ImportError message [(#336)](https://github.com/Clarifai/clarifai-python/pull/336)

## [[10.3.1]](https://github.com/Clarifai/clarifai-python/releases/tag/10.3.1) - [PyPI](https://pypi.org/project/clarifai/10.3.1/) - 2024-04-19

### Added
- Pagination feature in Search. Added *pagination* param in ```Search()``` class and included *per_page* and *page_no* params in ```Search().query()``` [(#331)](https://github.com/Clarifai/clarifai-python/pull/331)
- *Alogrithm* param in ```Search()```[(#331)](https://github.com/Clarifai/clarifai-python/pull/331)

### Changed
- Model Upload CLI Doc[(#329)](https://github.com/Clarifai/clarifai-python/pull/329)

### Fixed
- RAG.setup() bug where if we delete a specific workflow and create another workflow with the same id, by adding timestamp while creating a new prompter model [(#332)](https://github.com/Clarifai/clarifai-python/pull/332)
- ```RAG.upload()``` to support folder of text files.[(#332)](https://github.com/Clarifai/clarifai-python/pull/332)

## [[10.3.0]](https://github.com/Clarifai/clarifai-python/releases/tag/10.3.0) - [PyPI](https://pypi.org/project/clarifai/10.3.0/) - 2024-04-08

### Added
- Root certificate support to establish secure gRPC connections by adding ```root_certificates_path``` param in all the classes and auth helper and updating the grpc to the latest version.[(#319)](https://github.com/Clarifai/clarifai-python/pull/319)
- Missing VERSION and requirements.txt files to setup.py[(#320)](https://github.com/Clarifai/clarifai-python/pull/320)

### Changed
- To limit max upload batch size for ```Inputs().upload_inputs()``` function. Also changed the model version id parameter inconsistency in ```App.model()``` and ```Model()```[(#317)](https://github.com/Clarifai/clarifai-python/pull/317)

### Fixed
- Training status bug by removing constraint of user specifying *model_type_id* for training_logs and using ```load_info()``` to get model version details[(#321)](https://github.com/Clarifai/clarifai-python/pull/321)
- Create workflow bug which occured due to the model version id parameter change in #317[(#322)](https://github.com/Clarifai/clarifai-python/pull/322)
- Unnecessary infra alerts by adding wait time before deleting a model in model training tests [(#326)](https://github.com/Clarifai/clarifai-python/pull/326)

### Removed
- Runners from the SDK[(#325)](https://github.com/Clarifai/clarifai-python/pull/325)

## [[10.2.1]](https://github.com/Clarifai/clarifai-python/releases/tag/10.2.1) - [PyPI](https://pypi.org/project/clarifai/10.2.1/) - 2024-03-19

### Added
- Dataset version ID support in ```app.dataset()``` and ```Dataset()``` [(#315)](https://github.com/Clarifai/clarifai-python/pull/315)

### Changed
- Dataset Export function to internally download the dataset archive zip with the function ```Dataset.archive_zip()```[(#303)](https://github.com/Clarifai/clarifai-python/pull/303)
- The backoff iterator to support custom starting count, so different process can have different starting wait times.[(#313)](https://github.com/Clarifai/clarifai-python/pull/313)

### Fixed
- Removed the key *base_embed_model* from params.yaml file, since the model training by default considers the base embed model which is set for the app and no need to define it again in params file.[(#314)](https://github.com/Clarifai/clarifai-python/pull/314)

## [[10.2.0]](https://github.com/Clarifai/clarifai-python/releases/tag/10.2.0) - [PyPI](https://pypi.org/project/clarifai/10.2.0/) - 2024-03-18

### Added
- Model Export support [(#304)](https://github.com/Clarifai/clarifai-python/pull/304)
- Retry upload function from failed log files [(#307)](https://github.com/Clarifai/clarifai-python/pull/307)

### Fixed
- File not found error in model serving CLI [(#305)](https://github.com/Clarifai/clarifai-python/pull/305)
- Workflow YAML schema bug [(#308)](https://github.com/Clarifai/clarifai-python/pull/308)
- Base URL passing bug [(#308)](https://github.com/Clarifai/clarifai-python/pull/308)

## [[10.1.1]](https://github.com/Clarifai/clarifai-python/releases/tag/10.1.1) - [PyPI](https://pypi.org/project/clarifai/10.1.1/) - 2024-02-28

### Added
- Eval Endpoints (#290)
- Eval Utils (#296)
- Eval Tests (#297)
- Support session token (#300)

### Changed
- Dataset upload Enhancements (#292)

### Fixed
- Concept ID check befor model training (#295)
- RAG setup debug (#298)
- Requirements Update (#299)

## [[10.1.0]](https://github.com/Clarifai/clarifai-python/releases/tag/10.1.0) - [PyPI](https://pypi.org/project/clarifai/10.1.0/) - 2024-02-13

### Added
- Model Upload v2 CLI (#269)
- Support Existing App in RAG (#275)
- Support RAG Prompter kwargs (#280)
- Custom Workflow id support in RAG (#291)

### Fixed
- Model Template Change in Model Train Test (#273)
- Dataset Upload summary fix (#282)
- Update Model Serving Docs (#287)

## [[10.0.1]](https://github.com/Clarifai/clarifai-python/releases/tag/10.0.1) - [PyPI](https://pypi.org/project/clarifai/10.0.1/) - 2024-01-18

### Fixed
- Modified process_response_keys functions to fetch metadata info (#270)
- Assert user_id condition for RAG (#268)

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
