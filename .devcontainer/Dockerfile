FROM continuumio/anaconda3 as upstream

# Verify OS version is expected one
RUN . /etc/os-release && if [ "${VERSION_CODENAME}" != "bookworm" ]; then exit 1; fi

# Temporary: Upgrade python packages due to mentioned CVEs
# They are installed by the base image (continuumio/anaconda3) which does not have the patch.
COPY ./apply_security_patches.sh /tmp/apply_security_patches.sh
RUN chmod +x /tmp/apply_security_patches.sh
RUN /tmp/apply_security_patches.sh

# Reset and copy updated files with updated privs to keep image size down
FROM mcr.microsoft.com/devcontainers/base:1-bookworm

ARG USERNAME=vscode

# Create the conda group and add remote user to the group
RUN groupadd -r conda --gid 900 \ 
    && usermod -aG conda ${USERNAME}

# Copy opt folder, set ownership and group permissions
COPY --chown=:conda --chmod=775 --from=upstream /opt/conda /opt/conda
RUN chmod =2775 /opt/conda

USER root

# Copy scripts to execute
COPY add-notice.sh /tmp/library-scripts/

# Setup conda to mirror contents from https://github.com/ContinuumIO/docker-images/blob/master/anaconda3/debian/Dockerfile
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH
ARG USERNAME=vscode
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        libglib2.0-0 \
        libsm6 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxi6 \
        libxinerama1 \
        libxrandr2 \
        libxrender1 \
        mercurial \
        openssh-client \
        procps \
        subversion \
        wget \
    && apt-get upgrade -y \
    && bash /tmp/library-scripts/add-notice.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts/add-notice.sh

# Copy environment.yml (if found) to a temp location so we can update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
# COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
COPY environment.yml* noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp


RUN \
    apt-get update && export DEBIAN_FRONTEND=noninteractive && \
    apt-get -y install sudo cmake build-essential gfortran git openmpi-bin openmpi-common libopenmpi-dev libomp-dev

RUN \
    apt-get update && export DEBIAN_FRONTEND=noninteractive && apt install -y gpg-agent wget && \
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list && \
    apt update && \
    apt install -y intel-oneapi-mkl-devel

RUN \
    cd /opt && \
    git clone https://github.com/m-a-d-n-e-s-s/madness.git && \
    cd madness && \
    mkdir build && \
    cd build && \
    cmake -DENABLE_MPI=OFF -DENABLE_MKL=ON -DCMAKE_INSTALL_PREFIX=/usr/local/madness -DLAPACK_LIBRARIES=/opt/intel/oneapi/mkl/2025.0/lib/libmkl_rt.so -DLAPACK_INCLUDE_DIRS=/opt/intel/oneapi/mkl/2025.0/include/ -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" .. && \
    cmake --build . --target install -j2

RUN \
    pip install pyscf && \
    pip install tequila-basic && \
    pip install qulacs && \
    conda install -y madtequila -c kottmann && \
    pip install block2