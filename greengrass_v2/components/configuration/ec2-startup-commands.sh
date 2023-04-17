# if Ubuntu linux
sudo apt-get update && sudo apt-get -y upgrade
sudo apt-get install -y docker.io
sudo service docker start
sudo usermod -a -G docker ubuntu
sudo docker pull casualcodersm/iot_device_simulator
sudo docker run -it casualcodersm/iot_device_simulator
#Follow steps specified in ReadMe.md


# if Amazon Linux2
sudo yum update && sudo yum -y upgrade
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user
sudo systemctl enable docker.service
sudo systemctl start docker.service
sudo docker pull casualcodersm/iot_device_simulator
sudo docker run -it casualcodersm/iot_device_simulator
#Follow steps specified in ReadMe.md