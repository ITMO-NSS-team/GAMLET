# Download base image ubuntu 20.04
FROM ubuntu:20.04

# For apt to be noninteractive
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

# Preseed tzdata, update package index, upgrade packages and install needed software
RUN truncate -s0 /tmp/preseed.cfg; \
    echo "tzdata tzdata/Areas select Europe" >> /tmp/preseed.cfg; \
    echo "tzdata tzdata/Zones/Europe select Berlin" >> /tmp/preseed.cfg; \
    debconf-set-selections /tmp/preseed.cfg && \
    rm -f /etc/timezone /etc/localtime && \
	apt-get update && \
	apt-get install -y nano  && \
	apt-get install -y mc && \
    apt-get install -y python3.9 python3-pip && \
	apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Set the workdir
ENV WORKDIR /home/meta-automl-research
WORKDIR $WORKDIR
COPY . $WORKDIR

RUN pip3 install pip && \
    pip install wheel && \
    pip install --trusted-host pypi.python.org -r ${WORKDIR}/requirements.txt

ENV PYTHONPATH $WORKDIR
