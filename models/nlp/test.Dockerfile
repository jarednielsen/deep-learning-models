FROM nvidia/cuda:10.2-devel-ubuntu18.04

ENV NCCL_VERSION=2.1
ARG NCCL_VERSION=something

# Prints 2.7.8; changing ARG to ENV prints 'something'
RUN echo "nccl_version = ${NCCL_VERSION}"
