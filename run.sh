#!/bin/bash

python3 -m venv .venv

. .venv/bin/activate
python3 -m pip install -r requirements.txt

cd src

python3 main.py