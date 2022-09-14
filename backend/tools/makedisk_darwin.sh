#!/bin/bash
set -e
cd ${0%/*}/../../
rm -rf ./out | true
python -m nuitka \
    --standalone \
    --windows-disable-console \
    --lto=no \
    --include-data-dir=backend=backend \
    --include-data-dir=$CONDA_PREFIX/lib/python3.8/site-packages=dependencies \
    --include-data-dir=$CONDA_PREFIX/lib/python3.8/tkinter=dependencies/tkinter \
    --include-data-dir=$CONDA_PREFIX/lib/python3.8/email=dependencies/email \
    --include-data-dir=$CONDA_PREFIX/lib/python3.8/http=dependencies/http \
    --include-data-dir=$CONDA_PREFIX/lib/python3.8/distutils=dependencies/distutils \
    --include-data-dir=$CONDA_PREFIX/lib/python3.8/unittest=dependencies/unittest \
    --include-data-file=$CONDA_PREFIX/lib/python3.8/*.py=dependencies/ \
    --include-data-file=$CONDA_PREFIX/lib/python3.8/lib-dynload/_tkinter.cpython-38-darwin.so=dependencies/_tkinter.cpython-38-darwin.so \
    --nofollow-imports \
    --windows-icon-from-ico=./design/vse.ico \
    --plugin-enable=tk-inter,multiprocessing \
    --output-dir=out \
    ./gui.py 