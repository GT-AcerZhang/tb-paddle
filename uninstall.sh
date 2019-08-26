#!/bin/bash

set -ex

rm -rf build/ dist/ protoc/ tb_paddle.egg-info
rm -f protoc-3.6.1-osx-x86_64.zip
pip uninstall tb-paddle -y
