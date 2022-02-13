FROM python:3.9-slim
FROM nvidia/cuda:11.4.2-devel-ubuntu20.04
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel


RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN rm -rf /opt/algorithm/*
RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip
RUN pip install --upgrade pip


COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm entrypoint.sh /opt/algorithm/
COPY --chown=algorithm:algorithm configs /opt/algorithm/configs
# COPY --chown=algorithm:algorithm datasets /opt/algorithm/datasets
COPY --chown=algorithm:algorithm detectron2 /opt/algorithm/detectron2
COPY --chown=algorithm:algorithm detectron2.egg-info /opt/algorithm/detectron2.egg-info
COPY --chown=algorithm:algorithm docs /opt/algorithm/docs
COPY --chown=algorithm:algorithm tools /opt/algorithm/tools
COPY --chown=algorithm:algorithm projects /opt/algorithm/projects
COPY --chown=algorithm:algorithm coco_json.py /opt/algorithm/
COPY --chown=algorithm:algorithm coco_json_test.py /opt/algorithm/
COPY --chown=algorithm:algorithm postprocessing.py /opt/algorithm/
COPY --chown=algorithm:algorithm test_main.py /opt/algorithm/
COPY --chown=algorithm:algorithm process_launchers.py /opt/algorithm/
COPY --chown=algorithm:algorithm setup.py /opt/algorithm/
COPY --chown=algorithm:algorithm setup.cfg /opt/algorithm/
COPY --chown=algorithm:algorithm maskr_final_100.pth /opt/algorithm/
COPY --chown=algorithm:algorithm maskr_final_995.pth /opt/algorithm/
COPY --chown=algorithm:algorithm maskr_final_99.pth /opt/algorithm/
COPY --chown=algorithm:algorithm retina_final_100.pth /opt/algorithm/
COPY --chown=algorithm:algorithm retina_final_995.pth /opt/algorithm/
COPY --chown=algorithm:algorithm retina_final_99.pth /opt/algorithm/


# for training purpose only
#COPY --chown=algorithm:algorithm input /opt/algorithm/input

RUN python -m pip install --user -rrequirements.txt
RUN python -m pip install -e .

COPY --chown=algorithm:algorithm process.py /opt/algorithm/process.py

ENTRYPOINT ["bash", "entrypoint.sh"]

## ALGORITHM LABELS ##

# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=NodeDetectionMaskRCNNV1Container
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=4
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=2
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=256G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=4
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=4
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=40G
