FROM nvidia/cuda:10.1-devel-ubuntu18.04

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ENV HOROVOD_VERSION=0.19.5
ENV TENSORFLOW_PIP=tensorflow
ENV TENSORFLOW_VERSION=2.3.0
ENV TENSORFLOW_ADDONS_VERSION=0.11.1
ENV PYTORCH_VERSION=1.6.0
ENV TORCHVISION_VERSION=0.7.0
ENV CUDNN_VERSION=7.6.5.32-1+cuda10.1
ENV NCCL_VERSION=2.4.8-1+cuda10.1
ENV MXNET_VERSION=1.6.0.post0

# Python 3.6 is supported by Ubuntu Bionic out of the box
ARG python=3.7
ENV PYTHON_VERSION=${python}

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        g++-4.8 \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libcudnn7=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers \
        autoconf \
        libcupti-dev

##### EFA here ######

# ARG EFA_INSTALLER_VERSION=latest
# # ARG NCCL_VERSION=v2.6.4-1
# ARG AWS_OFI_NCCL_VERSION=aws
# ARG NCCL_TESTS_VERSION=v2.0.0

# ENV HOME /tmp

# ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:$HOME/nccl/build/lib:/opt/amazon/efa/lib:/$HOME/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH
# ENV PATH=/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:$PATH


#################################################
## Install EFA installer
# RUN cd $HOME \
#     && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
#     && tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
#     && cd aws-efa-installer \
#     && ./efa_installer.sh -y -d --skip-kmod --skip-limit-conf --no-verify | tee $HOME/install.log

# ## Test EFA installation
# RUN /opt/amazon/efa/bin/fi_info --version

###################################################
## Install NCCL
# RUN git clone https://github.com/NVIDIA/nccl.git $HOME/nccl \
#     && cd $HOME/nccl \
#     && git checkout ${NCCL_VERSION} \
#     && make -j src.build CUDA_HOME=/usr/local/cuda \
#     NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_60,code=sm_60"

###################################################
## Install AWS-OFI-NCCL plugin
# RUN git clone https://github.com/aws/aws-ofi-nccl.git $HOME/aws-ofi-nccl \
#     && cd $HOME/aws-ofi-nccl \
#     && apt-get update && apt-get install -y libudev-dev dh-autoreconf \
#     && git checkout ${AWS_OFI_NCCL_VERSION} \
#     && ./autogen.sh \
#     && ./configure --prefix=$HOME/aws-ofi-nccl/install \
#        --with-libfabric=/opt/amazon/efa/ \
#        --with-cuda=/usr/local/cuda \
#        --with-nccl=$HOME/nccl/build \
#        --with-mpi=/opt/amazon/openmpi/ \
#     && make && make install

# Download EFA NCCL adapter - Ben's version
# RUN cd / \
#     && git clone https://github.com/aws/aws-ofi-nccl.git -b aws \
#     && cd aws-ofi-nccl \
#     && apt-get update && apt-get install -y libudev-dev dh-autoreconf \
#     && ./autogen.sh \
#     && ./configure \
#        --with-libfabric=/efa \
#        --with-cuda=/usr/local/cuda \
#        --with-nccl=/nccl/build \
#        --with-mpi=/usr/local \
#        --prefix=/aws-ofi-nccl/install \
#     && make && make install

###################################################
## Install NCCL-tests
# RUN git clone https://github.com/NVIDIA/nccl-tests.git $HOME/nccl-tests \
#     && cd $HOME/nccl-tests \
#     && git checkout ${NCCL_TESTS_VERSION} \
#     && make MPI=1 \
#        MPI_HOME=/opt/amazon/openmpi/ \
#        CUDA_HOME=/usr/local/cuda \
#        NCCL_HOME=$HOME/nccl/build

# CMD echo "EFA Setup Complete! "



######################

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install TensorFlow, Keras, PyTorch and MXNet
RUN pip install future typing
RUN pip install numpy \
        ${TENSORFLOW_PIP}==${TENSORFLOW_VERSION} \
        keras \
        h5py

# RUN pip install https://download.pytorch.org/whl/cu101/torch-${PYTORCH_VERSION}-$(python -c "import wheel.pep425tags as w; print('-'.join(w.get_supported(None)[0][:-1]))")-linux_x86_64.whl \
#         https://download.pytorch.org/whl/cu101/torchvision-${TORCHVISION_VERSION}-$(python -c "import wheel.pep425tags as w; print('-'.join(w.get_supported(None)[0][:-1]))")-linux_x86_64.whl
RUN pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION}
RUN pip install mxnet-cu101==${MXNET_VERSION}

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=1 \
         pip install --no-cache-dir horovod==${HOROVOD_VERSION} && \
    ldconfig

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Download examples
RUN apt-get install -y --no-install-recommends subversion && \
    svn checkout https://github.com/horovod/horovod/trunk/examples && \
    rm -rf /examples/.svn

WORKDIR "/examples"

###### Modifications to horovod Dockerfile below
# tensorflow_addons is tightly coupled to TF version. TF 2.1 = 0.9.1, TF 2.2 = 0.10.0
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        scikit-learn==0.23.1 \
        wandb==0.9.1 \
        tensorboard_plugin_profile \
        tensorflow-addons==${TENSORFLOW_ADDONS_VERSION} \
        colorama==0.4.3 \
        pandas \
        apache_beam

ENV HDF5_USE_FILE_LOCKING "FALSE"

WORKDIR /fsx
CMD ["/bin/bash"]

###### Modifications specifically for SageMaker are below
# Install SSH on SageMaker machines
RUN apt-get install -y --no-install-recommends openssh-client openssh-server
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
RUN mkdir -p /root/.ssh/ && \
    mkdir -p /var/run/sshd && \
    ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
    cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
    printf "Host *\n  StrictHostKeyChecking no\n" >> /root/.ssh/config
RUN pip install --no-cache-dir \
    mpi4py==3.0.3 \
    sagemaker-training \
    git+https://github.com/huggingface/transformers.git@master \
    git+https://github.com/huggingface/nlp.git@703b761

# Install tokenizers from master source
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="$HOME/.cargo/bin:$PATH"
RUN git clone https://github.com/huggingface/tokenizers /root/tokenizers && \
    cd /root/tokenizers/bindings/python
RUN PATH="$HOME/.cargo/bin:$PATH" pip install setuptools_rust
RUN cd /root/tokenizers/bindings/python && PATH="$HOME/.cargo/bin:$PATH" python setup.py install

###### Modifications specifically for EC2 connected to FSx for Lustre are below
# When you use `docker run`, you'll need to run two commands manually:
# pip install -e /fsx/transformers
# These are done in the MPIJob launch script when using Kubernetes, but not for a shell.
