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

#Video Series
1. https://www.youtube.com/watch?v=rLswsgz77I0&list=PLQ530p80agO_IMi_hVpMvJFvun8NUuTyC
2. https://youtu.be/DcppAQ9ENvA

Extra:
1. https://youtu.be/bigdrPB_2o8
2. https://youtu.be/NOdXRAFEvDo
3. https://youtu.be/qC2U_dZfGCo