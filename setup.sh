#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REQ_FILE="${SCRIPT_DIR}/requirements.txt"

WEIGHTS_DIR="${SCRIPT_DIR}/weights"
WEIGHTS_FILE="${WEIGHTS_DIR}/model_final_bd324a.pkl"
WEIGHTS_URL="https://dl.fbaipublicfiles.com/detectron2/PanopticDeepLab/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32/model_final_bd324a.pkl"

CITYSCAPES_DIR="${SCRIPT_DIR}/data/cityscapes"

LEFTIMG_ZIP="${SCRIPT_DIR}/leftImg8bit_trainvaltest.zip"
GTFINE_ZIP="${SCRIPT_DIR}/gtFine_trainvaltest.zip"

echo "Installing Python requirements..."
python -m pip install -q -r "${REQ_FILE}"

echo "Installing Cityscapes scripts..."
python -m pip install -q git+https://github.com/mcordts/cityscapesScripts.git

echo "Preparing weights directory..."
mkdir -p "${WEIGHTS_DIR}"

if [ ! -f "${WEIGHTS_FILE}" ]; then
    echo "Downloading Panoptic-DeepLab weights..."
    curl -L "${WEIGHTS_URL}" -o "${WEIGHTS_FILE}"
fi

echo "Preparing Cityscapes dataset directory..."
mkdir -p "${CITYSCAPES_DIR}"

echo "Extracting Cityscapes datasets..."

unzip -q -o "${LEFTIMG_ZIP}" -d "${CITYSCAPES_DIR}"
unzip -q -o "${GTFINE_ZIP}" -d "${CITYSCAPES_DIR}"

export CITYSCAPES_DATASET="${CITYSCAPES_DIR}"

echo "Generating labelTrainIds..."
python -m cityscapesscripts.preparation.createTrainIdLabelImgs

echo "Cityscapes dataset path:"
echo "${CITYSCAPES_DIR}"

echo "Setup complete."