
https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/amazon-linux-ami-mate.html

# Steps for setting up IoT Device Simulator for Bearing Vibration Data
1. Pull Docker image from registry

docker pull 

---------------------------------------------------------------------------------------------------------------------------------
2. Run Containers - container specific

docker run --rm -e device_name="BearingAnomalySensor1" -e trained_model_dataset=1 --name BearingAnomalySensor1 -it casualcodersm/iot_device_simulator

docker run --rm -e device_name="BearingAnomalySensor2" -e trained_model_dataset=2 --name BearingAnomalySensor2 -it casualcodersm/iot_device_simulator

docker run --rm -e device_name="BearingAnomalySensor3" -e trained_model_dataset=3 --name BearingAnomalySensor3 -it casualcodersm/iot_device_simulator

docker run --rm -e device_name="BearingAnomalySensor4" -e trained_model_dataset=4 --name BearingAnomalySensor4 -it casualcodersm/iot_device_simulator

---------------------------------------------------------------------------------------------------------------------------------
3. Setup environment within container - common (Run in each container)

export AWS_ACCESS_KEY_ID=<access_key>
export AWS_SECRET_ACCESS_KEY=<secret_key>

curl -s https://d2s8p88vqu9w66.cloudfront.net/releases/greengrass-nucleus-latest.zip \
> greengrass-nucleus-latest.zip && unzip greengrass-nucleus-latest.zip -d GreengrassInstaller

OR

wget https://d2s8p88vqu9w66.cloudfront.net/releases/greengrass-nucleus-latest.zip greengrass-nucleus-latest.zip 
unzip greengrass-nucleus-latest.zip -d GreengrassInstaller


---------------------------------------------------------------------------------------------------------------------------------
4. Setup environment within container - container specific

sudo -E java -Droot="/greengrass/v2" -Dlog.store=FILE -jar ./GreengrassInstaller/lib/Greengrass.jar --aws-region ap-south-1 --thing-name Mimic-BearingAnomalySensor_1 --thing-group-name Mimic-BearingVibrationSensorsGroup --component-default-user ggc_user:ggc_group --provision true --setup-system-service true --deploy-dev-tools true

sudo -E java -Droot="/greengrass/v2" -Dlog.store=FILE -jar ./GreengrassInstaller/lib/Greengrass.jar --aws-region ap-south-1 --thing-name Mimic-BearingAnomalySensor_2 --thing-group-name Mimic-BearingVibrationSensorsGroup --component-default-user ggc_user:ggc_group --provision true --setup-system-service true --deploy-dev-tools true

sudo -E java -Droot="/greengrass/v2" -Dlog.store=FILE -jar ./GreengrassInstaller/lib/Greengrass.jar --aws-region ap-south-1 --thing-name Mimic-BearingAnomalySensor_3 --thing-group-name Mimic-BearingVibrationSensorsGroup --component-default-user ggc_user:ggc_group --provision true --setup-system-service true --deploy-dev-tools true

sudo -E java -Droot="/greengrass/v2" -Dlog.store=FILE -jar ./GreengrassInstaller/lib/Greengrass.jar --aws-region ap-south-1 --thing-name Mimic-BearingAnomalySensor_4 --thing-group-name Mimic-BearingVibrationSensorsGroup --component-default-user ggc_user:ggc_group --provision true --setup-system-service true --deploy-dev-tools true

---------------------------------------------------------------------------------------------------------------------------------
5. Continue Setup environment within container - common (Run in each container)

/greengrass/v2/alts/current/distro/bin/loader&
---------------------------------------------------------------------------------------------------------------------------------
6. Check the Core devices which shall list new device with healthy state
/greengrass/v2/bin/greengrass-cli component list
---------------------------------------------------------------------------------------------------------------------------------

# Steps for setting up IoT Device Simulator handling using AWS Greengrass v2 service at Cloud side
1. Setup S3 bucket for Artifacts and note it's arn. For example : arn:aws:s3:::bearing-anomaly-detector-project-bucket
Reference : 
https://docs.aws.amazon.com/greengrass/v2/developerguide/getting-started.html 
=> "Step 5: Create your component in the AWS IoT Greengrass service" => Create your component in AWS IoT Greengrass (console)
 

2. Create User Componenent 
Create component 'bearing_anomaly_detector' using the bucket arn:aws:s3:::bearing-anomaly-detector-project-bucket
Reference :
https://docs.aws.amazon.com/greengrass/v2/developerguide/getting-started.html 
=>Step 6: Deploy your component => Deploy your component (console)



# Additional greengrass commands at the IoT Device simulator

## Stoping/Pausing common IoT Device Services
/greengrass/v2/bin/greengrass-cli component stop -n=aws.greengrass.Nucleus
/greengrass/v2/bin/greengrass-cli component stop -n=DeploymentService
/greengrass/v2/bin/greengrass-cli component stop -n=UpdateSystemPolicyService
/greengrass/v2/bin/greengrass-cli component stop -n=FleetStatusService
/greengrass/v2/bin/greengrass-cli component stop -n=TelemetryAgent
/greengrass/v2/bin/greengrass-cli component stop -n=aws.greengrass.LocalDebugConsole
/greengrass/v2/bin/greengrass-cli component stop -n=aws.greengrass.Cli
        
## Restart/Resume common IoT Device Services
/greengrass/v2/bin/greengrass-cli component restart -n=aws.greengrass.Nucleus
/greengrass/v2/bin/greengrass-cli component restart -n=aws.greengrass.Cli
/greengrass/v2/bin/greengrass-cli component restart -n=DeploymentService
/greengrass/v2/bin/greengrass-cli component restart -n=UpdateSystemPolicyService
/greengrass/v2/bin/greengrass-cli component restart -n=FleetStatusService
/greengrass/v2/bin/greengrass-cli component restart -n=TelemetryAgent
/greengrass/v2/bin/greengrass-cli component restart -n=aws.greengrass.LocalDebugConsole

/greengrass/v2/bin/greengrass-cli component restart -n=bearing_vibration_anomaly_detection

## get Local Debug Console password
/greengrass/v2/bin/greengrass-cli get-debug-password
Change from default Configurration 
<code>
{
  "port": "10441",
  "httpsEnabled": "false",
  "websocketPort": "10442",
  "bindHostname": "0.0.0.0"
}
</code>