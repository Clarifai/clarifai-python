## [[11.6.7]](https://github.com/Clarifai/clarifai-python/releases/tag/11.6.7) - [PyPI](https://pypi.org/project/clarifai/11.6.7/) - 2025-08-04

### Changed
- skip code generation when context is None [(#730)] (https://github.com/Clarifai/clarifai-python/pull/730)
- pipeline_steps should be used in templates [(#728)] (https://github.com/Clarifai/clarifai-python/pull/728)
- Fix nodepool creation [(#729)] (https://github.com/Clarifai/clarifai-python/pull/729)
- Fix pipeline status code checks [(#727)] (https://github.com/Clarifai/clarifai-python/pull/727)
- various fixes for pipelines [(#726)] (https://github.com/Clarifai/clarifai-python/pull/726)
- Add list / ls CLI command for pipeline and pipelinestep [(#667)] (https://github.com/Clarifai/clarifai-python/pull/667)
- Fix PAT account settings link [(#724)] (https://github.com/Clarifai/clarifai-python/pull/724)

## [[11.6.6]](https://github.com/Clarifai/clarifai-python/releases/tag/11.6.6) - [PyPI](https://pypi.org/project/clarifai/11.6.6/) - 2025-07-30

### Changed
- Added support for verbose logging of Ollama [(#717)] (https://github.com/Clarifai/clarifai-python/pull/717)
- Improve error messages with pythonic models [(#721)] (https://github.com/Clarifai/clarifai-python/pull/721)
- Improve login logging experience [(#719)] (https://github.com/Clarifai/clarifai-python/pull/719)
- Improve Local Runner Logging [(#720)] (https://github.com/Clarifai/clarifai-python/pull/720)

## [[11.6.5]](https://github.com/Clarifai/clarifai-python/releases/tag/11.6.5) - [PyPI](https://pypi.org/project/clarifai/11.6.5/) - 2025-07-23

### Changed
- Add CLI config context support to BaseClient authentication [(#704)] (https://github.com/Clarifai/clarifai-python/pull/704)
- live logging functionality for model runner [(#711)] (https://github.com/Clarifai/clarifai-python/pull/711)
- Unify Context Management Under a Single config Command [(#709)] (https://github.com/Clarifai/clarifai-python/pull/709)
- Add func to return both stub and channel [(#713)] (https://github.com/Clarifai/clarifai-python/pull/713)
- Added local-runner requirements validation step [(#712)] (https://github.com/Clarifai/clarifai-python/pull/712)
- Improve URL Download error handling [(#710)] (https://github.com/Clarifai/clarifai-python/pull/710)
- Added Playground URL to Local-Runner Logs [(#708)] (https://github.com/Clarifai/clarifai-python/pull/708)
- Unit tests for toolkits [(#639)] (https://github.com/Clarifai/clarifai-python/pull/639)
- Improve Local-Runner CLI Logging [(#706)] (https://github.com/Clarifai/clarifai-python/pull/706)
- Improve client script formatting (black linter formatting) [(#705)] (https://github.com/Clarifai/clarifai-python/pull/705)
- Add github folder download support and toolkit option in model init [(#699)] (https://github.com/Clarifai/clarifai-python/pull/699)
- Improve Handling for PAT and USER_ID [(#702)] (https://github.com/Clarifai/clarifai-python/pull/702)

## [[11.6.4]](https://github.com/Clarifai/clarifai-python/releases/tag/11.6.4) - [PyPI](https://pypi.org/project/clarifai/11.6.4/) - 2025-07-11

### Changed
- Fixed flag for local runner threads, add user validation error [(#698)] (https://github.com/Clarifai/clarifai-python/pull/698)
- Added PAT token validation during clarifai login command [(#697)] (https://github.com/Clarifai/clarifai-python/pull/697)
- Fixed Local Runners Name across SDK [(#695)] (https://github.com/Clarifai/clarifai-python/pull/695)

## [[11.6.3]](https://github.com/Clarifai/clarifai-python/releases/tag/11.6.3) - [PyPI](https://pypi.org/project/clarifai/11.6.3/) - 2025-07-09

### Changed
- Added default template for ollama models in the local-runner ising `model init` command [(#693)] (https://github.com/Clarifai/clarifai-python/pull/693)
- Fixed `pipelinestep upload` command to parse all compute-info params and preserve user Dockerfile
- Fixed base model template import & return issues [(#690)] (https://github.com/Clarifai/clarifai-python/pull/690)
- Add `pool_size` flag default to 1 for local dev runner threads [(#689)] (https://github.com/Clarifai/clarifai-python/pull/689)

## [[11.6.2]](https://github.com/Clarifai/clarifai-python/releases/tag/11.6.2) - [PyPI](https://pypi.org/project/clarifai/11.6.2/) - 2025-07-08

### Changed
- Updated local-runner constants [(#684)] (https://github.com/Clarifai/clarifai-python/pull/684)

## [[11.6.1]](https://github.com/Clarifai/clarifai-python/releases/tag/11.6.1) - [PyPI](https://pypi.org/project/clarifai/11.6.1/) - 2025-07-07

### Changed
- Added `--version` flag support to the Clarifai CLI [(#678)] (https://github.com/Clarifai/clarifai-python/pull/678)
- Ensured better handling of `model_type_id` and improved configuration management [(#676)] (https://github.com/Clarifai/clarifai-python/pull/676)
- Added support for specifying a `deployment_user_id` in the Model class to enhance runner selection functionality [(#675)] (https://github.com/Clarifai/clarifai-python/pull/675)
- Added functionality to initialize a model directory from a GitHub repository, enhancing flexibility and usability in `model init` command [(#674)] (https://github.com/Clarifai/clarifai-python/pull/674)
- Fixed CLI PATH for Windows [(#672)] (https://github.com/Clarifai/clarifai-python/pull/672)
- Fixed code generation script [(#671)] (https://github.com/Clarifai/clarifai-python/pull/671)
- Added an alias for the pipelinestep CLI command and significantly improved test coverage for the `clarifai.runners.pipeline_steps` module [(#665)] (https://github.com/Clarifai/clarifai-python/pull/665)
- Improved CLI documentation and added descriptive help messages for various model-related commands [(#663)] (https://github.com/Clarifai/clarifai-python/pull/663)

## [[11.6.0]](https://github.com/Clarifai/clarifai-python/releases/tag/11.6.0) - [PyPI](https://pypi.org/project/clarifai/11.6.0/) - 2025-07-01

### Changed
- Number of threads used for GRPC Server default to CLARIFAI_NUM_THREADS and 32 otherwise [(#661)] (https://github.com/Clarifai/clarifai-python/pull/661)
- Use Configuration contexts in Model Upload CLI [(#649)] (https://github.com/Clarifai/clarifai-python/pull/649)
- Add pipeline run CLI similar to model predict [(#644)] (https://github.com/Clarifai/clarifai-python/pull/644)
- Update requirements.txt for protocol version [(#668)] (https://github.com/Clarifai/clarifai-python/pull/668)

## [[11.5.6]](https://github.com/Clarifai/clarifai-python/releases/tag/11.5.6) - [PyPI](https://pypi.org/project/clarifai/11.5.6/) - 2025-06-30

### Added
- Per-output token context tracking for batch operations
- New `set_output_context()` method for models to specify token usage per output

### Changed
- Improved token usage tracking in ModelClass with thread-local storage
- Enhanced batch processing support with ordered token context queue

### Fixed
- Token context ordering in batch operations using FIFO queue approach
- Temporarily disabled `test_client_batch_generate` while implementing token tracking features

## [[11.5.5]](https://github.com/Clarifai/clarifai-python/releases/tag/11.5.5) - [PyPI](https://pypi.org/project/clarifai/11.5.5/) - 2025-06-27

### Fixed
- fix legacy proto support [(#636)] (https://github.com/Clarifai/clarifai-python/pull/636)

## [[11.5.4]](https://github.com/Clarifai/clarifai-python/releases/tag/11.5.4) - [PyPI](https://pypi.org/project/clarifai/11.5.4/) - 2025-06-25

### Fixed
- Added authentication support to URL fetcher for SDH-protected URLs [(#647)] (https://github.com/Clarifai/clarifai-python/pull/647)

## [[11.5.3]](https://github.com/Clarifai/clarifai-python/releases/tag/11.5.3) - [PyPI](https://pypi.org/project/clarifai/11.5.3/) - 2025-06-24

### Changed
- Fixes AMD-related configuration by updating image versioning, introducing an AMD-specific Torch image  [(#641)] (https://github.com/Clarifai/clarifai-python/pull/641)
- Fix code snippets and Added code snippet test  [(#638)] (https://github.com/Clarifai/clarifai-python/pull/638)
- Add CLI command for pipeline upload with orchestration and validation  [(#634)] (https://github.com/Clarifai/clarifai-python/pull/634)
- Add list models information in CLI and method [(#640)] (https://github.com/Clarifai/clarifai-python/pull/640)
- Show a terminal prompt asking users if they want to create a new app when the specified app does not exist  [(#637)] (https://github.com/Clarifai/clarifai-python/pull/637)
- Asyncify predict endpoints v2  [(#588)] (https://github.com/Clarifai/clarifai-python/pull/588)
- Added Model Utils in SDK [(#631)] (https://github.com/Clarifai/clarifai-python/pull/631)
- Use model auth to set runner  [(#632)] (https://github.com/Clarifai/clarifai-python/pull/632)
- Add support for Clarifai Pipeline Steps Upload similar to Model Upload  [(#621)] (https://github.com/Clarifai/clarifai-python/pull/621)

## [[11.5.2]](https://github.com/Clarifai/clarifai-python/releases/tag/11.5.2) - [PyPI](https://pypi.org/project/clarifai/11.5.2/) - 2025-06-13

### Changed
- improve local dev and url helper [(630)](https://github.com/Clarifai/clarifai-python/pull/630)

### Fixed
- Proactively check code and requirements before upload [(625)](https://github.com/Clarifai/clarifai-python/pull/625)
- fix amd and circular imports [(628)](https://github.com/Clarifai/clarifai-python/pull/628)



## [[11.5.1]](https://github.com/Clarifai/clarifai-python/releases/tag/11.5.1) - [PyPI](https://pypi.org/project/clarifai/11.5.1/) - 2025-06-13

### Changed
- use uv in the build process [(626)](https://github.com/Clarifai/clarifai-python/pull/626)

### Fixed
- Proactively check code and requirements before upload [(625)](https://github.com/Clarifai/clarifai-python/pull/625)
- fix amd and circular imports [(628)](https://github.com/Clarifai/clarifai-python/pull/628)

## [[11.5.0]](https://github.com/Clarifai/clarifai-python/releases/tag/11.5.0) - [PyPI](https://pypi.org/project/clarifai/11.5.0/) - 2025-06-10

### Changed
- Removed an unused parameter in VisualClassifier class  [(#622)] (https://github.com/Clarifai/clarifai-python/pull/622)
- Add support to `/responses`, `/embeddings`, and `/images/generations` endpoints to the OpenAI class  [(#619)] (https://github.com/Clarifai/clarifai-python/pull/619)
- Fixed data display issue and updated openai params  [(#618)] (https://github.com/Clarifai/clarifai-python/pull/618)

## [[11.4.10]](https://github.com/Clarifai/clarifai-python/releases/tag/11.4.10) - [PyPI](https://pypi.org/project/clarifai/11.4.10/) - 2025-05-30

### Changed
- Add back in pretrained model config  [(#616)] (https://github.com/Clarifai/clarifai-python/pull/616)

## [[11.4.9]](https://github.com/Clarifai/clarifai-python/releases/tag/11.4.9) - [PyPI](https://pypi.org/project/clarifai/11.4.9/) - 2025-05-30

### Changed
- Updated Model Upload section in Readme  [(#613)] (https://github.com/Clarifai/clarifai-python/pull/613)
- Add clarifai model init to CLI to create default files for model upload [(#611)] (https://github.com/Clarifai/clarifai-python/pull/611)

## [[11.4.8]](https://github.com/Clarifai/clarifai-python/releases/tag/11.4.8) - [PyPI](https://pypi.org/project/clarifai/11.4.8/) - 2025-05-29

### Changed
- Fix issue with model upload  [(#612)] (https://github.com/Clarifai/clarifai-python/pull/612)


## [[11.4.7]](https://github.com/Clarifai/clarifai-python/releases/tag/11.4.7) - [PyPI](https://pypi.org/project/clarifai/11.4.7/) - 2025-05-29

### Changed
- Improve usage of clarifai config in urls  [(#608)] (https://github.com/Clarifai/clarifai-python/pull/608)
- Update code snippets for MCP / OpenAI  [(#607)] (https://github.com/Clarifai/clarifai-python/pull/607)


## [[11.4.6]](https://github.com/Clarifai/clarifai-python/releases/tag/11.4.6) - [PyPI](https://pypi.org/project/clarifai/11.4.6/) - 2025-05-28

### Changed
- Fixed Model Upload  [(#606)] (https://github.com/Clarifai/clarifai-python/pull/606)

## [[11.4.5]](https://github.com/Clarifai/clarifai-python/releases/tag/11.4.5) - [PyPI](https://pypi.org/project/clarifai/11.4.5/) - 2025-05-28

### Changed
- Fixed `MCPModelClass` notifications bug [(#602)] (https://github.com/Clarifai/clarifai-python/pull/602)
- Improved the `OpenAIModelClass` to streamline request processing, add modularity, and simplify parameter extraction and validation [(#601)] (https://github.com/Clarifai/clarifai-python/pull/601)
- Fixed a bug in the `OpenAIModelClass` to return the full json responses [(#597)] (https://github.com/Clarifai/clarifai-python/pull/597)
- Cleanup fastmcp [(#596)] (https://github.com/Clarifai/clarifai-python/pull/596)
- Added `OpenAIModelClass` to allow developers to create models that interact with OpenAI-compatible API endpoints [(#594)] (https://github.com/Clarifai/clarifai-python/pull/594)

## [[11.4.4]](https://github.com/Clarifai/clarifai-python/releases/tag/11.4.4) - [PyPI](https://pypi.org/project/clarifai/11.4.4/) - 2025-05-26

### Changed
- Fixed openai messages Utils function and code-snippet function [(#595)] (https://github.com/Clarifai/clarifai-python/pull/595)

## [[11.4.3]](https://github.com/Clarifai/clarifai-python/releases/tag/11.4.3) - [PyPI](https://pypi.org/project/clarifai/11.4.3/) - 2025-05-23

### Changed
- Simplified openai client wrapper functions [(#562)](https://github.com/Clarifai/clarifai-python/pull/562)
- MCP integration, CLI commands and improved environment variable handling [(#592)](https://github.com/Clarifai/clarifai-python/pull/592)

## [[11.4.2]](https://github.com/Clarifai/clarifai-python/releases/tag/11.4.2) - [PyPI](https://pypi.org/project/clarifai/11.4.2/) - 2025-05-21

### Changed
- Fix Pythonic bugs [(#586)](https://github.com/Clarifai/clarifai-python/pull/586)
- Addition of Base Class for Visual Classifier Models [(#585)](https://github.com/Clarifai/clarifai-python/pull/585)
- Print script after model upload [(#583)](https://github.com/Clarifai/clarifai-python/pull/583)
- Add AMD changes [(#581)](https://github.com/Clarifai/clarifai-python/pull/581)
- Removed duplicate model downloads and improved error logging for gated HF repo. [(#564)](https://github.com/Clarifai/clarifai-python/pull/564)
- Addition of Base Class for Visual Detector Models [(#563)](https://github.com/Clarifai/clarifai-python/pull/563)
- remove rich from req [(#560)](https://github.com/Clarifai/clarifai-python/pull/560)

## [[11.4.1]](https://github.com/Clarifai/clarifai-python/releases/tag/11.4.1) - [PyPI](https://pypi.org/project/clarifai/11.4.1/) - 2025-05-09

### Changed
- Param for Inference params in model.py and FE [(#567)](https://github.com/Clarifai/clarifai-python/pull/567)

## [[11.4.0]](https://github.com/Clarifai/clarifai-python/releases/tag/11.4.0) - [PyPI](https://pypi.org/project/clarifai/11.4.0/) - 2025-05-08

### Changed
- Fixed Streamlit Query Parameters retrieval issue in ClarifaiAuthHelper. [(#577)](https://github.com/Clarifai/clarifai-python/pull/577)
- Fixed pyproject.toml. [(#575)](https://github.com/Clarifai/clarifai-python/pull/575)
- Fixed local dev runners. [(#574)](https://github.com/Clarifai/clarifai-python/pull/574)
- Fixed issue of runner ID of local dev runners. [(#573)](https://github.com/Clarifai/clarifai-python/pull/573)
- Switched to `uv` and `ruff` to speed up tests and formatting & linting. [(#572)](https://github.com/Clarifai/clarifai-python/pull/572)
- Changed some `==` to `is`. [(#570)](https://github.com/Clarifai/clarifai-python/pull/570)
- Local dev runner setup using CLI is easier now. [(#568)](https://github.com/Clarifai/clarifai-python/pull/568)
- Fixed indirect inheritence from ModelClass. [(#566)](https://github.com/Clarifai/clarifai-python/pull/566)

## [[11.3.0]](https://github.com/Clarifai/clarifai-python/releases/tag/11.3.0) - [PyPI](https://pypi.org/project/clarifai/11.3.0/) - 2025-04-22

### Changed
- We support pythonic models now. See [runners-examples](https://github.com/clarifai/runners-examples) [(#525)](https://github.com/Clarifai/clarifai-python/pull/525)
- Fixed failing tests. [(#559)](https://github.com/Clarifai/clarifai-python/pull/559)

## [[11.2.3]](https://github.com/Clarifai/clarifai-python/releases/tag/11.2.3) - [PyPI](https://pypi.org/project/clarifai/11.2.3/) - 2025-04-08

### Changed
 - CLI is now abougt 20x faster for most operations [(#555)](https://github.com/Clarifai/clarifai-python/pull/555)
 - CLI now has config contexts, more to come there... [(#552)](https://github.com/Clarifai/clarifai-python/pull/552)
 - Improve error messages with missing PAT [(#548)](https://github.com/Clarifai/clarifai-python/pull/548)
 - Fix model builder return args [(#547)](https://github.com/Clarifai/clarifai-python/pull/547)

## [[11.2.2]](https://github.com/Clarifai/clarifai-python/releases/tag/11.2.2) - [PyPI](https://pypi.org/project/clarifai/11.2.2/) - 2025-03-28

### Changed
 - Removed HF loader `config.json` validation for all clarifai Model type ids [(#543)] (https://github.com/Clarifai/clarifai-python/pull/543)
 - Added Regex Patterns to Filter Checkpoint Files to Download [(#542)] (https://github.com/Clarifai/clarifai-python/pull/542)

## [[11.2.1]](https://github.com/Clarifai/clarifai-python/releases/tag/11.2.1) - [PyPI](https://pypi.org/project/clarifai/11.2.1/) - 2025-03-25

### Changed
 - Added validation for CLI config [(#540)] (https://github.com/Clarifai/clarifai-python/pull/540)
 - Fixed docker image name and added `skip_dockerfile` option to `test-locally` subcommand od model CLI [(#526)] (https://github.com/Clarifai/clarifai-python/pull/526)

## [[11.2.0]](https://github.com/Clarifai/clarifai-python/releases/tag/11.2.0) - [PyPI](https://pypi.org/project/clarifai/11.2.0/) - 2025-03-24

### Changed
 - Improved CLI login module [(#535)] (https://github.com/Clarifai/clarifai-python/pull/535)
 - Updated the CLI to test out model locally independent of remote access [(#534)] (https://github.com/Clarifai/clarifai-python/pull/534)
 - Modified the default value of `num_threads` field [(#533)] (https://github.com/Clarifai/clarifai-python/pull/533)

## [[11.1.7]](https://github.com/Clarifai/clarifai-python/releases/tag/11.1.7) - [PyPI](https://pypi.org/project/clarifai/11.1.7/) - 2025-03-08

### Changed
 - Dropped testing of python 3.8, 3.9, 3.10 [(#532)] (https://github.com/Clarifai/clarifai-python/pull/532)
 - Updated the deployment testing config [(#531)] (https://github.com/Clarifai/clarifai-python/pull/531)

## [[11.1.6]](https://github.com/Clarifai/clarifai-python/releases/tag/11.1.6) - [PyPI](https://pypi.org/project/clarifai/11.1.6/) - 2025-03-06

### Changed
 - Removed the model_path argument to CLI [(#529)] (https://github.com/Clarifai/clarifai-python/pull/529)
 - Added configuration for multi-threaded runners [(#524)] (https://github.com/Clarifai/clarifai-python/pull/524)

## [[11.1.5]](https://github.com/Clarifai/clarifai-python/releases/tag/11.1.5) - [PyPI](https://pypi.org/project/clarifai/11.1.5/) - 2025-02-21

### Changed
 - Adds support for local dev runners from CLI  [(#521)] (https://github.com/Clarifai/clarifai-python/pull/521)
 - Use the non-runtime path for tests [(#520)] (https://github.com/Clarifai/clarifai-python/pull/520)
 - Fix local tests [(#518)] (https://github.com/Clarifai/clarifai-python/pull/518)
 - Catch additional codes that models have at startup [(#517)] (https://github.com/Clarifai/clarifai-python/pull/517)

## [[11.1.4]](https://github.com/Clarifai/clarifai-python/releases/tag/11.1.4) - [PyPI](https://pypi.org/project/clarifai/11.1.4/) - 2025-02-12

### Changed

 - Introduce 3 times when you can download checkpoints [(#515)] (https://github.com/Clarifai/clarifai-python/pull/515)

## [[11.1.3]](https://github.com/Clarifai/clarifai-python/releases/tag/11.1.3) - [PyPI](https://pypi.org/project/clarifai/11.1.3/) - 2025-02-11

### Changed

 - Fix dependency parsing [(#514)] (https://github.com/Clarifai/clarifai-python/pull/514)

## [[11.1.2]](https://github.com/Clarifai/clarifai-python/releases/tag/11.1.2) - [PyPI](https://pypi.org/project/clarifai/11.1.2/) - 2025-02-10

### Changed

 - User new base images and fix clarifai version [(#513)] (https://github.com/Clarifai/clarifai-python/pull/513)

## [[11.1.1]](https://github.com/Clarifai/clarifai-python/releases/tag/11.1.1) - [PyPI](https://pypi.org/project/clarifai/11.1.1/) - 2025-02-06

### Changed

 - Don't validate API in server.py [(#509)] (https://github.com/Clarifai/clarifai-python/pull/509)

## [[11.1.0]](https://github.com/Clarifai/clarifai-python/releases/tag/11.1.0) - [PyPI](https://pypi.org/project/clarifai/11.1.0/) - 2025-02-05

### Changed

 - Fixed Docker test locally [(#505)] (https://github.com/Clarifai/clarifai-python/pull/505)
 - Fixed HF checkpoints error [(#504)] (https://github.com/Clarifai/clarifai-python/pull/504)
 - Fixed Deployment Tests [(#502)] (https://github.com/Clarifai/clarifai-python/pull/502)
 - Fixed Issue with Filename as Invalid Input ID [(#501)] (https://github.com/Clarifai/clarifai-python/pull/501)
 - Update Model Predict CLI [(#500)] (https://github.com/Clarifai/clarifai-python/pull/500)
 - Tests Health Port to None [(#499)] (https://github.com/Clarifai/clarifai-python/pull/499)
 - Refactor model class and runners to be more independent [(#494)] (https://github.com/Clarifai/clarifai-python/pull/494)
 - Add storage request inferred from tar and checkpoint size [(#479)] (https://github.com/Clarifai/clarifai-python/pull/479)

## [[11.0.7]](https://github.com/Clarifai/clarifai-python/releases/tag/11.0.7) - [PyPI](https://pypi.org/project/clarifai/11.0.7/) - 2025-01-24

### Changed

 - Updated model upload experience [(#498)] (https://github.com/Clarifai/clarifai-python/pull/498)

## [[11.0.6]](https://github.com/Clarifai/clarifai-python/releases/tag/11.0.6) - [PyPI](https://pypi.org/project/clarifai/11.0.6/) - 2025-01-24

### Changed

 - Added Model Upload Tests [(#495)] (https://github.com/Clarifai/clarifai-python/pull/495)
 - Updated Torch version Images and Delete tar file for every upload [(#493)] (https://github.com/Clarifai/clarifai-python/pull/493)
 - Added Tests for Model run locally [(#492)] (https://github.com/Clarifai/clarifai-python/pull/492)
 - Added CLARIFAI_API_BASE in the test container [(#491)] (https://github.com/Clarifai/clarifai-python/pull/491)
 - remove triton requirements [(#490)] (https://github.com/Clarifai/clarifai-python/pull/490)

## [[11.0.5]](https://github.com/Clarifai/clarifai-python/releases/tag/11.0.5) - [PyPI](https://pypi.org/project/clarifai/11.0.5/) - 2025-01-17

### Changed

 - Added tests for downloads and various improvements [(#489)] (https://github.com/Clarifai/clarifai-python/pull/489)

## [[11.0.4]](https://github.com/Clarifai/clarifai-python/releases/tag/11.0.4) - [PyPI](https://pypi.org/project/clarifai/11.0.4/) - 2025-01-17

### Changed

 - Added tests for downloads and various improvements [(#488)] (https://github.com/Clarifai/clarifai-python/pull/488)

## [[11.0.3]](https://github.com/Clarifai/clarifai-python/releases/tag/11.0.3) - [PyPI](https://pypi.org/project/clarifai/11.0.3/) - 2025-01-14

### Changed

 - Make API validation optional [(#483)] (https://github.com/Clarifai/clarifai-python/pull/483)
 - Env var to control logging [(#482)] (https://github.com/Clarifai/clarifai-python/pull/482)


## [[11.0.2]](https://github.com/Clarifai/clarifai-python/releases/tag/11.0.2) - [PyPI](https://pypi.org/project/clarifai/11.0.2/) - 2025-01-14

### Changed

 - Update base images [(#481)] (https://github.com/Clarifai/clarifai-python/pull/481)
 - Optimize downloads from HF [(#480)] (https://github.com/Clarifai/clarifai-python/pull/480)

## [[11.0.1]](https://github.com/Clarifai/clarifai-python/releases/tag/11.0.1) - [PyPI](https://pypi.org/project/clarifai/11.0.1/) - 2025-01-14

### Changed

 - Take user_id from Env variable [(#477)] (https://github.com/Clarifai/clarifai-python/pull/477)
 - Added HF token Validation [(#476)] (https://github.com/Clarifai/clarifai-python/pull/476)
 - Fix Model prediction methods when configured with a dedicated compute_cluster_id and nodepool_id [(#475)] (https://github.com/Clarifai/clarifai-python/pull/475)
 - Fix model upload issues [(#474)] (https://github.com/Clarifai/clarifai-python/pull/474)
 - Improved error logging [(#473)] (https://github.com/Clarifai/clarifai-python/pull/473)

## [[11.0.0]](https://github.com/Clarifai/clarifai-python/releases/tag/11.0.0) - [PyPI](https://pypi.org/project/clarifai/11.0.0/) - 2025-01-07

### Changed
 - Changed labels to optional in Dataloaders to support Data Ingestion pipelines in clarifai-datautils library [(#471)] (https://github.com/Clarifai/clarifai-python/pull/471)


## [[10.11.1]](https://github.com/Clarifai/clarifai-python/releases/tag/10.11.1) - [PyPI](https://pypi.org/project/clarifai/10.11.1/) - 2024-12-20

### Added
 - Added model building logs [(#467)] (https://github.com/Clarifai/clarifai-python/pull/467)
 - Added user_id to RAG class [(#466)] (https://github.com/Clarifai/clarifai-python/pull/466)
 - Added Compute Orchestration to README.md [(#461)] (https://github.com/Clarifai/clarifai-python/pull/461)
 - Added Testing and Running a model locally within a container [(#460)] (https://github.com/Clarifai/clarifai-python/pull/460)
 - Added CLI support for Model Predict [(#459)] (https://github.com/Clarifai/clarifai-python/pull/459)

### Changed
 - Updated Dockerfile for Sglang [(#468)] (https://github.com/Clarifai/clarifai-python/pull/468)
 - Updated available torch images and some refactoring [(#465)] (https://github.com/Clarifai/clarifai-python/pull/465)

### Fixed
 - Fixed issue for Model local testing [(#469)] (https://github.com/Clarifai/clarifai-python/pull/469)

### Removed
 - Removed protobuf from requirements to resolve conflicts with clarifai-grpc [(#464)] (https://github.com/Clarifai/clarifai-python/pull/464)

## [[10.11.0]](https://github.com/Clarifai/clarifai-python/releases/tag/10.11.0) - [PyPI](https://pypi.org/project/clarifai/10.11.0/) - 2024-12-03

### Changed
 - Fixed  issue of bounding box info edge cases [(#457)] (https://github.com/Clarifai/clarifai-python/pull/457)
 - Supports downloading data.parts as bytes [(#456)] (https://github.com/Clarifai/clarifai-python/pull/456)
 - Changed default env to prod for Model upload [(#455)] (https://github.com/Clarifai/clarifai-python/pull/455)
 - Added tests for all stream and generate methods [(#452)] (https://github.com/Clarifai/clarifai-python/pull/452)
 - Added Codecoverage test report in PRs [(#450)] (https://github.com/Clarifai/clarifai-python/pull/450)

## [[10.10.1]](https://github.com/Clarifai/clarifai-python/releases/tag/10.10.1) - [PyPI](https://pypi.org/project/clarifai/10.10.1/) - 2024-11-18

### Changed
 - Fixed code bug in runners selection using Deployment [(#446)] (https://github.com/Clarifai/clarifai-python/pull/446)
 - Fixed id bug in multimodal loader during deletion of failed inputs [(#445)] (https://github.com/Clarifai/clarifai-python/pull/445)
 - Added list inputs functionality to Dataset Class [(#443)] (https://github.com/Clarifai/clarifai-python/pull/443)
 - Added delete annotations functionality to Input Class [(#442)] (https://github.com/Clarifai/clarifai-python/pull/442)
 - Added Dockerle template based on new base images by parsing requirements [(#439)] (https://github.com/Clarifai/clarifai-python/pull/439)

## [[10.10.0]](https://github.com/Clarifai/clarifai-python/releases/tag/10.10.0) - [PyPI](https://pypi.org/project/clarifai/10.10.0/) - 2024-11-07

### Changed
 - Added a check for base url parameter [(#438)] (https://github.com/Clarifai/clarifai-python/pull/438)
 - Added CLI support for Compute Orchestration resources (Compute cluster, Nodepool, Deployment) [(#436)] (https://github.com/Clarifai/clarifai-python/pull/436)

## [[10.9.5]](https://github.com/Clarifai/clarifai-python/releases/tag/10.9.5) - [PyPI](https://pypi.org/project/clarifai/10.9.5/) - 2024-10-29

## [[10.9.4]](https://github.com/Clarifai/clarifai-python/releases/tag/10.9.4) - [PyPI](https://pypi.org/project/clarifai/10.9.4/) - 2024-10-28

## [[10.9.3]](https://github.com/Clarifai/clarifai-python/releases/tag/10.9.3) - [PyPI](https://pypi.org/project/clarifai/10.9.3/) - 2024-10-28

### Changed
 - Added tests for CRUD Operations of CO Resource - Deployment [(#431)] (https://github.com/Clarifai/clarifai-python/pull/431)
 - Added request-id-prefix header to SDK requests to improve SDK monitoring [(#430)] (https://github.com/Clarifai/clarifai-python/pull/430)
 - Added CLI for Model upload [(#429)] (https://github.com/Clarifai/clarifai-python/pull/429)
 - Fixed model servicer for Model Upload [(#428)] (https://github.com/Clarifai/clarifai-python/pull/428)
 - Added python versions badge to README.md [(#427)] (https://github.com/Clarifai/clarifai-python/pull/427)
 - Removed stream tests till stream API is fixed [(#426)] (https://github.com/Clarifai/clarifai-python/pull/426)
 - Removed unnecessary prefixes to concept ID added from SDK [(#424)] (https://github.com/Clarifai/clarifai-python/pull/424)
 - Upgraded llama-index-core lib version as a security update [(#423)] (https://github.com/Clarifai/clarifai-python/pull/423)
 - Added metadata in exported dataset annotations files
 - Upgrade to clarifai-grpc 10.9.11

## [[10.9.2]](https://github.com/Clarifai/clarifai-python/releases/tag/10.9.2) - [PyPI](https://pypi.org/project/clarifai/10.9.2/) - 2024-10-14

### Changed
 - Improve UX for model upload and fix runners tests [(#420)] (https://github.com/Clarifai/clarifai-python/pull/420)
 - Added functionality to Merge Datasets [(#419)] (https://github.com/Clarifai/clarifai-python/pull/419)
 - Fix bugs for model upload  [(#417)] (https://github.com/Clarifai/clarifai-python/pull/417)
 - Fix download_checkpoints and fix run model locally  [(#415)] (https://github.com/Clarifai/clarifai-python/pull/415)

## [[10.9.1]](https://github.com/Clarifai/clarifai-python/releases/tag/10.9.1) - [PyPI](https://pypi.org/project/clarifai/10.9.1/) - 2024-10-09

### Changed
 - Improve handling missing huggingface_hub package [(#412)] (https://github.com/Clarifai/clarifai-python/pull/412)
 - Implement script that allows users to test and run a runner's model locally [(#411)] (https://github.com/Clarifai/clarifai-python/pull/411)
 - Improve Model upload experience for cv models [(#408)] (https://github.com/Clarifai/clarifai-python/pull/408)

## [[10.9.0]](https://github.com/Clarifai/clarifai-python/releases/tag/10.9.0) - [PyPI](https://pypi.org/project/clarifai/10.9.0/) - 2024-10-07

### Changed
 - Improved the Test Coverage for Dataloaders & Evaluations modules of SDK [(#409)] (https://github.com/Clarifai/clarifai-python/pull/409)

## [[10.8.9]](https://github.com/Clarifai/clarifai-python/releases/tag/10.8.9) - [PyPI](https://pypi.org/project/clarifai/10.8.9/) - 2024-10-03

### Changed
 - New streaming predict endpoints[(#407)](https://github.com/Clarifai/clarifai-python/pull/407)
 - New dockerfile for model upload and improvements to upload flow[(#406)](https://github.com/Clarifai/clarifai-python/pull/406)

## [[10.8.8]](https://github.com/Clarifai/clarifai-python/releases/tag/10.8.8) - [PyPI](https://pypi.org/project/clarifai/10.8.8/) - 2024-10-03

### Changed
 - Bug fixes for logger[(#405)](https://github.com/Clarifai/clarifai-python/pull/405)

## [[10.8.7]](https://github.com/Clarifai/clarifai-python/releases/tag/10.8.7) - [PyPI](https://pypi.org/project/clarifai/10.8.7/) - 2024-10-03

### Added
 - Added CRUD operations for Compute Orchestration resources (Compute cluster, Nodepool, Deployment) [(#402)] (https://github.com/Clarifai/clarifai-python/pull/402)

### Changed
 - Improved logging and fixed issues with downloading checkpoints[(#403)](https://github.com/Clarifai/clarifai-python/pull/403)

## [[10.8.6]](https://github.com/Clarifai/clarifai-python/releases/tag/10.8.6) - [PyPI](https://pypi.org/project/clarifai/10.8.6/) - 2024-09-28

### Changed
 - Refract Model upload and download checkpoints at build time during model upload[(#400)](https://github.com/Clarifai/clarifai-python/pull/400)

## [[10.8.5]](https://github.com/Clarifai/clarifai-python/releases/tag/10.8.5) - [PyPI](https://pypi.org/project/clarifai/10.8.5/) - 2024-09-26

### Changed
 - Added fsspec dependency which would be required in runners for model upload [(#398)](https://github.com/Clarifai/clarifai-python/pull/398)
 - Added MultiModalLoader support [(#384)](https://github.com/Clarifai/clarifai-python/pull/384)
 - Deleted model_serving in this SDK, after the Runners PR has been merged [(#391)](https://github.com/Clarifai/clarifai-python/pull/391)

## [[10.8.4]](https://github.com/Clarifai/clarifai-python/releases/tag/10.8.4) - [PyPI](https://pypi.org/project/clarifai/10.8.4/) - 2024-09-26

### Changed
 - Added validation check in HF loader, if the checkpoints really exit at checkpoint path [(#396)](https://github.com/Clarifai/clarifai-python/pull/396)
 - Remove pydantic dependency from runners in clarifai-python [(#395)](https://github.com/Clarifai/clarifai-python/pull/395)

## [[10.8.3]](https://github.com/Clarifai/clarifai-python/releases/tag/10.8.3) - [PyPI](https://pypi.org/project/clarifai/10.8.3/) - 2024-09-25

### Changed
 - use json logger always in k8s [(#393)](https://github.com/Clarifai/clarifai-python/pull/393)
 - Added a json logger so it's convenient to get logs into logging stacks [(#392)](https://github.com/Clarifai/clarifai-python/pull/392)
 - Added HuggingFaceLoader and added methods in model_upload for download_checkpoints and handling concepts [(#390)](https://github.com/Clarifai/clarifai-python/pull/390)
 - Integrate clarifai-protocol which use to upload model to platform [(#389)](https://github.com/Clarifai/clarifai-python/pull/389)
 - Tests Addition for App, Dataset, Input, Model Classes [(#386)](https://github.com/Clarifai/clarifai-python/pull/386)
 - Upgrade to clarifai-grpc 10.8.7


## [[10.8.2]](https://github.com/Clarifai/clarifai-python/releases/tag/10.8.2) - [PyPI](https://pypi.org/project/clarifai/10.8.2/) - 2024-09-19

### Changed
 - Upgrade to clarifai-grpc 10.8.6
 - Improved Model Export functionality by adding `Ranges` header [(#385)](https://github.com/Clarifai/clarifai-python/pull/385)

## [[10.8.1]](https://github.com/Clarifai/clarifai-python/releases/tag/10.8.1) - [PyPI](https://pypi.org/project/clarifai/10.8.1/) - 2024-09-06

### Fixed
 - Python SDK usage issue on Windows OS due to upgrade in Protobuf library [(#380)](https://github.com/Clarifai/clarifai-python/pull/380)
 - Dataset Annotations bug that returns None if class annotation is not present during export [(#382)](https://github.com/Clarifai/clarifai-python/pull/382)

## [[10.8.0]](https://github.com/Clarifai/clarifai-python/releases/tag/10.8.0) - [PyPI](https://pypi.org/project/clarifai/10.8.0/) - 2024-09-02

### Added
 - Patch operations for Models and Workflows [(#370)] (https://github.com/Clarifai/clarifai-python/pull/370)
 - Addition of Concept Relations Operations [(#371)] (https://github.com/Clarifai/clarifai-python/pull/371)
 - Addition of App's Input Count functionality [(#372)] (https://github.com/Clarifai/clarifai-python/pull/372)

### Fixed
 - Dataset Annotations bug that returns either class annotation or detection annotation during export [(#375)](https://github.com/Clarifai/clarifai-python/pull/375)
 - Model Export Bug by adding authentication headers [(#373)](https://github.com/Clarifai/clarifai-python/pull/373)

## [[10.7.0]](https://github.com/Clarifai/clarifai-python/releases/tag/10.7.0) - [PyPI](https://pypi.org/project/clarifai/10.7.0/) - 2024-08-06

### Added
 - Patch operations for Apps and Datasets [(#364)] (https://github.com/Clarifai/clarifai-python/pull/364)

### Fixed
 - RAG class to support env variable for `user_id` param [(#357)](https://github.com/Clarifai/clarifai-python/pull/357)
 - Search query bug that returns duplicated triplets by removing `PostAnnotationsSearches` and replacing it with `PostInputsSearches`[(#366)](https://github.com/Clarifai/clarifai-python/pull/366)
 - Search request potentially blocks the users to use different types of filters altogether, fixed it by supporting annotation and input proto filters.[(#366)](https://github.com/Clarifai/clarifai-python/pull/366)

## [[10.5.4]](https://github.com/Clarifai/clarifai-python/releases/tag/10.5.4) - [PyPI](https://pypi.org/project/clarifai/10.5.4/) - 2024-07-12

### Added
 - Patch operations for input annotations and concepts [(#354)] (https://github.com/Clarifai/clarifai-python/pull/354)

### Changed
- Getting user id from ENV variables for RAG class [(#358)](https://github.com/Clarifai/clarifai-python/pull/358)
- Improved rich logging by width addition [(#359)](https://github.com/Clarifai/clarifai-python/pull/359)

### Fixed
 - Dataset export functionality - Added authentication headers to download requests, better exception formatting [(#356)](https://github.com/Clarifai/clarifai-python/pull/356)

## [[10.5.3]](https://github.com/Clarifai/clarifai-python/releases/tag/10.5.3) - [PyPI](https://pypi.org/project/clarifai/10.5.3/) - 2024-06-20

### Changed
- Moved some convenience features to CLI only to avoid writes to disk  [(#353)](https://github.com/Clarifai/clarifai-python/pull/353)

## [[10.5.2]](https://github.com/Clarifai/clarifai-python/releases/tag/10.5.2) - [PyPI](https://pypi.org/project/clarifai/10.5.2/) - 2024-06-20

### Changed
- Text Features to add random ID as input if input ID is not provided in Dataloader  [(#351)](https://github.com/Clarifai/clarifai-python/pull/351)

## [[10.5.1]](https://github.com/Clarifai/clarifai-python/releases/tag/10.5.1) - [PyPI](https://pypi.org/project/clarifai/10.5.1/) - 2024-06-17

### Changed
- Added BaseClient.from_env() and some new endpoints  [(#346)](https://github.com/Clarifai/clarifai-python/pull/346)

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
