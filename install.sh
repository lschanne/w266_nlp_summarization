#!/bin/bash

python3 -m venv venv_w266_final
source venv_w266_final/bin/activate

pip install --upgrade pip
pip install wheel
pip install --upgrade setuptools
#python -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-2.6.0-py3-none-any.whl
pip install -r requirements.txt

python ./download_data.py
