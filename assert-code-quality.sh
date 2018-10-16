#!/bin/bash
set -e

FIND_SOURCE_FILES='find clarifai tests -name "*.py" -not -path "clarifai/rest/grpc/proto/*"'

echo "START assert-code-quality.sh"
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

echo ""
echo "- isort: Make sure all imports are sorted"
# This checks only and ignores whitespace which is crucial because yapf defines the formatting.
isort -sp .isort.cfg -ws -c $(eval ${FIND_SOURCE_FILES})

if [ $? != 0 ]; then
  echo ""
  echo "  isort failed. Run the isort command without -c to automatically sort the imports correctly"
  echo "    Note: The import code style itself (besides the order) must still comply to yapf"
  exit 1
fi

echo "  Done isort"

echo ""
echo "- yapf: Make sure there are no code style issues"
yapf --style=.style.yapf -p -d $(eval ${FIND_SOURCE_FILES})

if [ $? != 0 ]; then
  echo ""
  echo "  yapf failed"
  exit 1
fi

echo "  Done yapf"
echo "SUCCESSFUL FINISH OF assert-code-quality.sh"
