#!/bin/bash

set -eo pipefail

FIND_SOURCE_FILES='find clarifai_utils tests -name "*.py"'

function run_autoflake {
  echo "- autoflake: Make sure there are no unused imports and unused variables"
  FIND_SOURCE_FILES_FOR_AUTOFLAKE="${FIND_SOURCE_FILES} -not -iwholename '*/__init__.py'"
  autoflake --remove-all-unused-imports --remove-unused-variables \
          $(eval ${FIND_SOURCE_FILES_FOR_AUTOFLAKE}) \
          | tee autoflake-output.tmp

  if [ $(cat autoflake-output.tmp | wc -c) != 0 ]; then
    echo ""
    echo "  Here are affected files: "
    grep "+++" autoflake-output.tmp
    rm autoflake-output.tmp
    echo "  autoflake failed"
    exit 1
  fi

  rm autoflake-output.tmp
  echo "  Done autoflake"
}


function run_yapf() {
  echo ""
  echo "- yapf: Make sure all the code is formatted consistently"

  yapf -i $(eval ${FIND_SOURCE_FILES})

  if [ $? != 0 ]; then
    echo ""
    echo "  yapf failed. Run './lint.sh' to automatically apply code formatting."
    exit 1
  fi

  echo "  Done yapf"
}


run_autoflake
run_yapf $1

echo "SUCCESSFUL FINISH OF lint.sh"
