#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import os
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

# Dynamically compile protos
def compileProtoBuf():
    res = subprocess.call(['bash', './compile.sh'])
    assert res == 0, 'cannot compile protobuf'

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        compileProtoBuf()
        develop.run(self)


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        compileProtoBuf()
        import os
        os.system("pip install protobuf numpy six crc32c moviepy")
        install.run(self)


with open('HISTORY.rst') as history_file:
    history = history_file.read()

preparing_PyPI_package = False
version = '0.1'

if not preparing_PyPI_package:
    with open('tb_paddle/__init__.py', 'a') as f:
        f.write('\n__version__ = "{}"\n'.format(version))

requirements = [
    'numpy',
    'protobuf >= 3.2.0',
    'six',
    'crc32c',
    'moviepy',
]

setup(
    name='tb_paddle',
    version=version,
    description='tb_paddle is a toolkit to visualize Paddle logged data in TensorBoard.',
    long_description=history,
    author='ShuLiang Lin',
    author_email='',
    url='https://github.com/linshuliang/TensorBoardPaddle',
    packages=['tb_paddle'],
    include_package_data=True,
    install_requires=requirements,
    license='MIT license',
    zip_safe=False,
    classifiers=[
        'Development Status :: 0.1',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)