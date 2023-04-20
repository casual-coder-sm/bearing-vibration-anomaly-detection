rm -r models/__pycache__ models/autoencoder_lstm/__pycache__
zip -r models.zip models
sha256sum models.zip
chdir greengrass_v2
zip -r ../scripts_mimic_iot.zip scripts_mimic_iot
chdir ..
sha256sum scripts_mimic_iot.zip
sha256sum greengrass_v2/components/artifacts/bearing_vibration_anomaly_detection/1.0.0/begin_mimic_devicesimulation.sh