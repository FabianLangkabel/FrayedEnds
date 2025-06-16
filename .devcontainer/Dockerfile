FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH

ARG USERNAME=vscode

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    wget \
    curl \
    git \
    bzip2 \
    ca-certificates \
    build-essential \
    gfortran \
    cmake \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    libomp-dev \
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
    procps \
    subversion \
    gnupg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneapi.list && \
    apt-get update && apt-get install -y intel-oneapi-mkl-devel && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN cd /opt && \
    git clone https://github.com/m-a-d-n-e-s-s/madness.git && \
    cd madness && \
    #git checkout 3aaceb6cbc9d2a283232e8e1335eacfea15a7e51 && \
    sed -i 's/testgconv.cc//' /opt/madness/src/madness/mra/CMakeLists.txt && \
    mkdir build && cd build && \
    cmake -DENABLE_MPI=OFF -DENABLE_MKL=ON \
        -DCMAKE_INSTALL_PREFIX=/usr/local/madness \
        -DLAPACK_LIBRARIES=/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so \
        -DLAPACK_INCLUDE_DIRS=/opt/intel/oneapi/mkl/latest/include/ \
        -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" .. && \
    cmake --build . --target install -j2

RUN conda install -y python=3.11 && \
    pip install pyscf && \
    pip install tequila-basic && \
    pip install qulacs && \
    pip install block2 && \
    python -m pip install nanobind

RUN conda install -c conda-forge nwchem && \
    conda install -c conda-forge openmpi && \
    conda install ipykernel --update-deps --force-reinstall

ENV MADNESS_DIR=/usr/local/madness
COPY . /workspaces/MRA-OrbitalOptimization
WORKDIR /workspaces/MRA-OrbitalOptimization

RUN pip install -e .

RUN groupadd --gid 900 conda && \
    useradd -m -s /bin/bash -u 1000 -g conda ${USERNAME} && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

ENV LD_LIBRARY_PATH="/opt/intel/oneapi/mkl/2025.1/lib:${LD_LIBRARY_PATH}"

USER ${USERNAME}
WORKDIR /home/${USERNAME}