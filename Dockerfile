FROM siderust:latest

ENV SIDERUST_LAB_ROOT="/home/user/src/"

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils software-properties-common \
    autoconf automake libtool build-essential git sudo cmake g++ \
    && cmake --version \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://git.code.sf.net/p/libnova/libnova /tmp/libnova && \
    cd /tmp/libnova && \
    ./autogen.sh && \
    ./configure && \
    make && \
    sudo make install

RUN pip install astropy pandas matplotlib

RUN useradd -m "lab" -s /usr/bin/bash
USER "lab"

CMD ["bash"]
