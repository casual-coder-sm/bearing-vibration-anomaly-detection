echo "#######################################"
echo "Received Message ${1}"
echo "#######################################"

comp_ver=${1}
rm -Rf /root/models
unzip -o /greengrass/v2/packages/artifacts/bearing_vibration_anomaly_detection/${1}/models.zip -d /root
#chdir /root/models
#conda run --no-capture-output -n cds python autoencoder_lstm/autoencoder_lstm_pred.py ${1} 1 2

unzip -o /greengrass/v2/packages/artifacts/bearing_vibration_anomaly_detection/${1}/scripts_mimic_iot.zip -d /root
chdir /root/scripts_mimic_iot
conda run --no-capture-output -n cds python ./publish_anomaly_status.py ${1} 1 2 100

