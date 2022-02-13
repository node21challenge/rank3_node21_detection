#!/usr/bin/env bash

./build.sh

docker save nodedetectionmaskrcnnv1container | gzip -c > NodeDetectionMaskRCNNV1Container.tar.gz
