# Buiding Container with AWS GreengrassV2

# Deploying Sample program 
## Deploy using device (local)
sudo /greengrass/v2/bin/greengrass-cli deployment create \
  --recipeDir ~/greengrassv2/components/recipes \
  --artifactDir ~/greengrassv2/components/artifacts \
  --merge "com.example.HelloWorld=1.0.0"

. View log
sudo tail -f /greengrass/v2/logs/greengrass.log

. Deploy and Restart after any modification to python file
sudo /greengrass/v2/bin/greengrass-cli component restart \
  --names "com.example.HelloWorld"

. Check deployed component
sudo /greengrass/v2/bin/greengrass-cli component list

. Remove after testing
sudo /greengrass/v2/bin/greengrass-cli deployment create --remove="com.example.HelloWorld"

. Stop component
 /greengrass/v2/bin/greengrass-cli component stop -n=bearing_anomaly_detector

 . Restart
  /greengrass/v2/bin/greengrass-cli component restart -n=bearing_anomaly_detector 

## Deploy using console 
aws iam create-policy \
  --policy-name gg-hello-world-artifact-policy \
  --policy-document file://component-artifact-policy.json



## Communication At Local