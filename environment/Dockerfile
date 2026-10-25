FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/kuroname
COPY environment/requirements-docker.txt /tmp/requirements-docker.txt
RUN python -m pip install --no-cache-dir --no-deps --break-system-packages -r /tmp/requirements-docker.txt && rm /tmp/requirements-docker.txt
COPY core /opt/kuroname/core
COPY artifacts /opt/kuroname/artifacts
COPY configs /opt/kuroname/configs
COPY src /opt/kuroname/src
COPY inference.sh /opt/kuroname/inference.sh
COPY environment/image_README.md /opt/kuroname/README.md
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0 NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
CMD ["bash"]
