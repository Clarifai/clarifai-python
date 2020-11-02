## [[2.6.2]](https://github.com/Clarifai/clarifai-python/releases/tag/2.6.2) - [PyPI](https://pypi.org/project/clarifai/2.6.2/) - 2019-07-16

### Changed
- Model Version is applied in get_info

### Fixed
- Construct Face only if region face is present


## [[2.6.1]](https://github.com/Clarifai/clarifai-python/releases/tag/2.6.1) - [PyPI](https://pypi.org/project/clarifai/2.6.1/) - 2019-03-11

### Added
- Support for Protobuf version 3.7


## [[2.6.0]](https://github.com/Clarifai/clarifai-python/releases/tag/2.6.0) - [PyPI](https://pypi.org/project/clarifai/2.6.0/) - 2019-02-28

### Added
- Type hinting

### Fixed
- Serialization of the predict request


## [[2.5.2]](https://github.com/Clarifai/clarifai-python/releases/tag/2.5.2) - [PyPI](https://pypi.org/project/clarifai/2.5.2/) - 2019-01-14

### Fixed
- Apply crop when searching by an image


## [[2.5.1]](https://github.com/Clarifai/clarifai-python/releases/tag/2.5.1) - [PyPI](https://pypi.org/project/clarifai/2.5.1/) - 2019-01-08

### Fixed
- Reintroduced requests session usage


## [[2.5.0]](https://github.com/Clarifai/clarifai-python/releases/tag/2.5.0) - [PyPI](https://pypi.org/project/clarifai/2.5.0/) - 2018-12-10

### Added
- Sample milliseconds parameter for video prediction
- Make tag_* methods deprecated

### Fixed
- Rename the embed model used in predict_embed


## [[2.4.2]](https://github.com/Clarifai/clarifai-python/releases/tag/2.4.2) - [PyPI](https://pypi.org/project/clarifai/2.4.2/) - 2018-11-20

### Fixed
- Fix setting error object fields on ApiError


## [[2.4.1]](https://github.com/Clarifai/clarifai-python/releases/tag/2.4.1) - [PyPI](https://pypi.org/project/clarifai/2.4.1/) - 2018-11-08

### Fixed
- Ignore unknown fields in the response object


## [[2.4.0]](https://github.com/Clarifai/clarifai-python/releases/tag/2.4.0) - [PyPI](https://pypi.org/project/clarifai/2.4.0/) - 2018-10-16

### Added
- Moderation solution

### Fixed
- Raise ApiError instead of requests.ConnectionError when URL not accessible


## [[2.3.2]](https://github.com/Clarifai/clarifai-python/releases/tag/2.3.2) - [PyPI](https://pypi.org/project/clarifai/2.3.2/) - 2018-09-25

### Fixed
- Fix how Protobuf is used to allow simultaneous use of other Protobuf libraries


## [[2.3.1]](https://github.com/Clarifai/clarifai-python/releases/tag/2.3.1) - [PyPI](https://pypi.org/project/clarifai/2.3.1/) - 2018-08-29

### Fixed
- Changed the location of the generated proto files to avoid a namespace conflict that appeared in certain cases


## [[2.3.0]](https://github.com/Clarifai/clarifai-python/releases/tag/2.3.0) - [PyPI](https://pypi.org/project/clarifai/2.3.0/) - 2018-08-20

### Changed
- Use Protobuf/gRPC to help with request/response JSON (de)serialization
- Add `assert-code-quality.sh` for static analysis and run it on Travis, fix code style
- Add AppVeyor

### Removed

- Remove unsupported patch outputs and get outputs on inputs


## [[2.2.3]](https://github.com/Clarifai/clarifai-python/releases/tag/2.2.3) - [PyPI](https://pypi.org/project/clarifai/2.2.3/) - 2018-05-10

### Changed
- Use requests.Session to persist and retry connections


## [[2.2.2]](https://github.com/Clarifai/clarifai-python/releases/tag/2.2.2) - [PyPI](https://pypi.org/project/clarifai/2.2.2/) - 2018-05-03

### Fixed
- Installing dependencies on Python <3.4


## [[2.2.1]](https://github.com/Clarifai/clarifai-python/releases/tag/2.2.1) - [PyPI](https://pypi.org/project/clarifai/2.2.1/) - 2018-04-25

### Fixed
- A dependency version


## [[2.2.0]](https://github.com/Clarifai/clarifai-python/releases/tag/2.2.0) - [PyPI](https://pypi.org/project/clarifai/2.2.0/) - 2018-04-25

### Added
- More tests

### Changed
- Made dependency versions ranges instead of using specific versions

### Removed
- V1 API support (including Pillow dependency)
- Removed checking for upgrades on every client initialization


## [[2.1.0]](https://github.com/Clarifai/clarifai-python/releases/tag/2.1.0) - [PyPI](https://pypi.org/project/clarifai/2.1.0/)

### Added
- Added support for custom face recognition
- Added public models 'enum' object
- Started using semantic versioning


## [2.0.32] - [PyPI](https://pypi.org/project/clarifai/2.0.32/)

### Added
- workflow predict support
- API_KEY initialization and doc fix


## [2.0.31] - [PyPI](https://pypi.org/project/clarifai/2.0.31/)

### Added
- retries on all 500 errors
- add a lib dependency


## [2.0.30] - [PyPI](https://pypi.org/project/clarifai/2.0.30/)

### Added
- expose ModelOutputConfig and ModelOutputInfo
- add search feedback api
- doc fox for feedback api

### Changed
- refactor feedback api

### Fixed
- fragile test fix
- slash model_id support fix


## [2.0.29] - [PyPI](https://pypi.org/project/clarifai/2.0.29/)

### Fixed
- hotfix for base_url regression


## [2.0.28] - [PyPI](https://pypi.org/project/clarifai/2.0.28/)

### Added
- min_value support
- max_concepts support
- select_concepts support
- feedback support for concept, detection and face detection
- nosetest --with-timer
- code examples for new features
- one-off test code for new features
- doc update and fix from fixit week

### Fixed
- auth key related bug fix
- other minor bugfixs


## [2.0.27] - [PyPI](https://pypi.org/project/clarifai/2.0.27/)

### Fixed
- inputs.get_all() mixed status fix
- inputs.get_by_page() mixed status fix


## [2.0.26] - [PyPI](https://pypi.org/project/clarifai/2.0.26/)

### Changed
- verbose in Exception
- less verbose in testing
- strip the url in predict and search


## [2.0.25] - [PyPI](https://pypi.org/project/clarifai/2.0.25/)

### Fixed
- bugfix for the api key handling for permission restricted users


## [2.0.24] - [PyPI](https://pypi.org/project/clarifai/2.0.24/)

### Fixed
- bugfix for still broken pip3 install


## [2.0.23] - [PyPI](https://pypi.org/project/clarifai/2.0.23/)

### Fixed
- bugfix for setup.py for pip3


## [2.0.22] - [PyPI](https://pypi.org/project/clarifai/2.0.22/)

### Added
- hyper parameters
- patch concepts
- video prediction support

### Fixed
- fix client code bugs
- fix github blocking issue for upgrade
- fix repeatedly predict with Image() bug
- speed up nosetests
- fix multiple concurrent test bugs


## [2.0.21] - [PyPI](https://pypi.org/project/clarifai/2.0.21/)

### Added
- tutorial for installation on Windows, to address issue #0
- retry on API gateway timeout
- back off on throttled API

### Fixed
- bugfix for `clarifai diagnose` on Windows
- other bug and doc fixes


## [2.0.20] - [PyPI](https://pypi.org/project/clarifai/2.0.20/)

### Added
- propagate the error with more informative text
- expose search score

### Fixed
- bugfix for the token expiration bug


## [2.0.19] - [PyPI](https://pypi.org/project/clarifai/2.0.19/)

### Added
- get input outputs
- patch input outputs
- shorten the base64 in the test logs
- backoff polling in the sync training

### Fixed
- bugfix on unexpected warnings


## [2.0.18] - [PyPI](https://pypi.org/project/clarifai/2.0.18/)

### Added
- multi-language support
- test coverage boost


## [2.0.14] - [PyPI](https://pypi.org/project/clarifai/2.0.14/)

### Added
- python3 support
- add simplified tagging calls

### Changed
- voluntary resizing in v1 call
- update docs


## [2.0.13] - [PyPI](https://pypi.org/project/clarifai/2.0.13/)

### Added
- more tests for metadata patch
- doc example for metadata patch

### Fixed
- fix the verbose debug info


## [2.0.12] - [PyPI](https://pypi.org/project/clarifai/2.0.12/)

### Added
- generalize PATCH for models
- generalize PATCH for inputs
- code refactoring for line width
- more tests
- tutorial for making Image()
- prettier debug info
- remove unused code
- timeout for training
- support filename in Image()

### Fixed
- fix a bug in Python3
- fix Windows compatibility issue
