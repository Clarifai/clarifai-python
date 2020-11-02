#!/bin/bash

version=$(grep 'version' setup.py | grep -o '[0-9].*[0-9].*[0-9]')

release_tag="v${version}"

git tag $release_tag
git push origin $release_tag

