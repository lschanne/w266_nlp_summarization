#!/bin/bash

python3 -m venv venv_w266_final
source venv_w266_final/bin/activate
pip install -r requirements.txt

python ./download_data.py
