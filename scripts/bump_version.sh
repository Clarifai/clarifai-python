#!/bin/bash

version=$(grep 'version' setup.py | grep -o '[0-9].*[0-9].*[0-9]')
base_version=${version%.*}
release_version=${version##*.}
next_release_version=$((release_version + 1))
next_version="${base_version}.${next_release_version}"

version_sed="$(echo $version | sed 's/\./\\./g')"
next_version_sed="$(echo $next_version | sed 's/\./\\./g')"

echo "Current version: $version"
echo "Next version:    $next_version"

for file in docs/conf.py setup.py clarifai/versions.py
do
  sed -i "s/${version_sed}/${next_version_sed}/g" $file
done
