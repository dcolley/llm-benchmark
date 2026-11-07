# NGC PyTorch 26.07 (torch 2.13) arm64, intended for GB10 / sm_121.
# Pin by digest so the same image can be used on both training machines.
FROM nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c

WORKDIR /work

# Apt versions are pinned in apt-packages.txt and recorded again after install.
COPY apt-packages.txt /tmp/apt-packages.txt
RUN apt-get update \
    && grep -vE '^#|^$' /tmp/apt-packages.txt > /tmp/apt-packages.pins \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        $(cat /tmp/apt-packages.pins) \
    && dpkg-query -W -f='${Package}=${Version}\n' ca-certificates curl git unzip \
        | tee /opt/text8-repro-apt-versions.txt \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /work

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Default command is scoring, not training.
CMD ["python", "score.py", "--help"]
