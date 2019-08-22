#!/bin/bash

set -ex

rm -rf build/ dist/ protoc/ tb-paddle.egg-info
rm -f protoc-3.6.1-osx-x86_64.zip
rm -f tb_paddle/proto/*_pb2.py
pip uninstall tb-paddle -y

