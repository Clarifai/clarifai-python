Contributing
============

#. Fork & clone this repo.

#. Make the changes.

#. Make sure all the code quality checks and unit/integration tests pass.


Running Code Quality Checks
---------------------------

1. Make sure the checks pass:

    `./assert-code-quality.sh`


Running Tests
-------------

To successfully run integration tests, you have to have a valid Clarifai API key with all required permissions.

Create a new API key at the `API keys page <https://www.clarifai.com/developer/account/keys>`_ and set it as an environmental variable `CLARIFAI_API_KEY`.

    Warning: The requests made by integration tests are run against the production system and will use your operations.


#. Set your Clarifai API key:

    `export CLARIFAI_API_KEY=your_clarifai_api_key`

#. Run the tests:

    `pytest -n auto`
