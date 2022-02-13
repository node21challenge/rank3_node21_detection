#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

docker volume create nodedetectionmaskrcnnv1container-output

docker run --rm \
        --gpus all \
        --shm-size=50g \
        -v $SCRIPTPATH/test/:/input/ \
        -v nodedetectionmaskrcnnv1container-output:/output/ \
        nodedetectionmaskrcnnv1container

docker run --rm \
        -v nodedetectionmaskrcnnv1container-output:/output/ \
        python:3.9-slim cat /output/nodules.json | python3 -m json.tool


docker run --rm \
        -v nodedetectionmaskrcnnv1container-output:/output/ \
        -v $SCRIPTPATH/test/:/input/ \
        python:3.9-slim python -c "import json, sys; f1 = json.load(open('/output/nodules.json')); f2 = json.load(open('/input/expected_output.json')); sys.exit(f1 != f2);"

if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi
