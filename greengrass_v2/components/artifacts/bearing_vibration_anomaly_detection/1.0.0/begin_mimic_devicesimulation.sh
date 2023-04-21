#!/bin/bash

echo "#######################################"
echo "Shell Arguments"
echo "Component Version : ${1}"
echo "Deployment  Type  : ${2}"
echo "IMS Dataset ID    : ${3}"
echo "Step Size         : ${4}"
echo "#######################################"

rm -Rf /root/models
unzip -o /greengrass/v2/packages/artifacts/bearing_vibration_anomaly_detection/${1}/models.zip -d /root
unzip -o /greengrass/v2/packages/artifacts/bearing_vibration_anomaly_detection/${1}/scripts_mimic_iot.zip -d /root
chdir /root/scripts_mimic_iot
conda run --no-capture-output -n cds python ./publish_anomaly_status.py ${1} ${2} ${3} ${4}

