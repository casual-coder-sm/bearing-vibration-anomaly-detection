#!/bin/bash

echo "#######################################"
echo "Shell Arguments"
echo "Component Version : ${1}"
echo "Deployment  Type  : ${2}"
echo "IMS Dataset ID    : ${3}"
echo "Dataset step size : ${4}"
echo "Predict step size : ${5}"
echo "#######################################"

rm -Rf /root/models
unzip -o /greengrass/v2/packages/artifacts/bearing_vibration_anomaly_detection/${1}/models.zip -d /root
unzip -o /greengrass/v2/packages/artifacts/bearing_vibration_anomaly_detection/${1}/scripts_mimic_iot.zip -d /root
chdir /root/models
conda run --no-capture-output -n cds python /root/scripts_mimic_iot/publish_anomaly_status.py ${1} ${2} ${3} ${4} ${5}

