#!/usr/bin/env python3

import argparse
import copy
import json
import sys

CUDA_VERSIONS_DICT = {
    "nightly": ["cu121"],
    "test": ["cu121", "cu124"],
    "release": ["cu121", "cu124"],
}

PYTHON_VERSIONS_DICT = {
    "nightly": ["3.9"],
    "test": ["3.9", "3.12"],
    "release": ["3.9", "3.12"],
}

TENSORRT_VERSIONS_DICT = {
    "windows": {
        "10.4": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.4.0/zip/TensorRT-10.4.0.26.Windows.win10.cuda-12.6.zip",
            "strip_prefix": "TensorRT-10.4.0.26",
            "sha256": "3a7de83778b9e9f812fd8901e07e0d7d6fc54ce633fcff2e340f994df2c6356c",
        },
        "10.5": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.5.0/zip/TensorRT-10.5.0.18.Windows.win10.cuda-12.6.zip",
            "strip_prefix": "TensorRT-10.5.0.18",
            "sha256": "e6436f4164db4e44d727354dccf7d93755efb70d6fbfd6fa95bdfeb2e7331b24",
        },
    },
    "linux": {
        "10.4": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.4.0/tars/TensorRT-10.4.0.26.Linux.x86_64-gnu.cuda-12.6.tar.gz",
            "strip_prefix": "TensorRT-10.4.0.26",
            "sha256": "cb0273ecb3ba4db8993a408eedd354712301a6c7f20704c52cdf9f78aa97bbdb",
        },
        "10.5": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.5.0/tars/TensorRT-10.5.0.18.Linux.x86_64-gnu.cuda-12.6.tar.gz",
            "strip_prefix": "TensorRT-10.5.0.18",
            "sha256": "f404d379d639552a3e026cd5267213bd6df18a4eb899d6e47815bbdb34854958",
        },
    },
}


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        help="matrix",
        type=str,
        default="",
    )

    options = parser.parse_args(args)
    if options.matrix == "":
        raise Exception(f"--matrix is empty, please provide the matrix json str")

    matrix_dict = json.loads(options.matrix)
    includes = matrix_dict["include"]
    assert len(includes) > 0
    channel = includes[0].channel
    if "windows" in includes[0].validation_runner:
        arch = "windows"
    elif "linux" in includes[0].validation_runner:
        arch = "linux"
    else:
        raise Exception(
            f"{includes[0].validation_runner} is not the supported arch, currently only support windows and linux"
        )

    cuda_versions = CUDA_VERSIONS_DICT[channel]
    python_versions = PYTHON_VERSIONS_DICT[channel]
    tensorrt_versions = TENSORRT_VERSIONS_DICT[arch]

    filtered_includes = []
    for item in includes:
        if (
            item["desired_cuda"] in cuda_versions
            and item["python_version"] in python_versions
        ):
            for _, tensorrt_json in tensorrt_versions.items():
                new_item = copy.deepcopy(item)
                new_item["tensorrt"] = tensorrt_json
                filtered_includes.append(new_item)
    filtered_matrix_dict = {}
    filtered_matrix_dict["include"] = filtered_includes
    print(json.dumps(filtered_matrix_dict))


if __name__ == "__main__":
    main(sys.argv[1:])
