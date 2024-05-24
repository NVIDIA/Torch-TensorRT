#!/usr/bin/env bash
set -eou pipefail
# Source conda so it's available to the script environment
source ${BUILD_ENV_FILE}
${CONDA_RUN} ${PIP_INSTALL_TORCH} torchvision
${CONDA_RUN} python -m pip install pyyaml mpmath==1.3.0
export TRT_VERSION=$(${CONDA_RUN} python -c "import versions; versions.tensorrt_version()")

# Install TensorRT manually
wget -q -P /opt/torch-tensorrt-builds/ https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz
tar -xzf /opt/torch-tensorrt-builds/TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz -C /opt/torch-tensorrt-builds/
python -m pip install /opt/torch-tensorrt-builds/TensorRT-10.0.1.6/python/tensorrt-10.0.1-cp${PYTHON_VERSION//./}-none-linux_x86_64.whl

# Install Torch-TensorRT
${CONDA_RUN} python -m pip install /opt/torch-tensorrt-builds/torch_tensorrt*+${CU_VERSION}*.whl

echo -e "Running test script";
