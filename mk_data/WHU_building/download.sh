#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Define save directory
SAVEDIR="$(pwd)/data"
mkdir -p "${SAVEDIR}"

# Download the ZIP file
wget -P "${SAVEDIR}" \
    "https://gpcv.whu.edu.cn/data/3.%20The%20cropped%20aerial%20image%20tiles%20and%20raster%20labels.zip"

# Unzip the downloaded file
ZIPFILE="${SAVEDIR}/3. The cropped aerial image tiles and raster labels.zip"
unzip -q "${ZIPFILE}" -d "${SAVEDIR}"

# Rename the extracted directory to WHU_building
EXTRACTED_DIR=$(find "${SAVEDIR}" -mindepth 1 -maxdepth 1 -type d ! -name "WHU_building")
mv "${EXTRACTED_DIR}" "${SAVEDIR}/WHU_building"

# Rename label directories as specified
mv "${SAVEDIR}/WHU_building/train/label" "${SAVEDIR}/WHU_building/train/clean_mask"
mv "${SAVEDIR}/WHU_building/val/label"   "${SAVEDIR}/WHU_building/val/mask"
mv "${SAVEDIR}/WHU_building/test/label"  "${SAVEDIR}/WHU_building/test/mask"

echo "Dataset prepared at: ${SAVEDIR}/WHU_building"
