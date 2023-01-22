FROM tensorflow/tensorflow:latest-gpu

# mount the current directory to /workspace
WORKDIR /workspace

COPY requirements.txt /workspace

# Install Python and its tools
RUN apt update && apt install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools

RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["jupyter", "lab", "--port=8080", "--no-browser", "--ip=0.0.0.0", "--allow-root"]