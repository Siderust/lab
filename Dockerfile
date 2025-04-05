FROM siderust:latest

ENV SIDERUST_LAB_ROOT="/home/user/src/"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf automake libtool \ 
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install python src & test dependencies
RUN pip install astropy pandas matplotlib

# Default command: open a shell for interaction
CMD ["bash"]
