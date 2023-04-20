## install miniconda

#1. Download miniconda from following location: 
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

#2. Install Miniconda (ensure to activate conda at the end):
sh ./Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

#3. Add to environment (Assuming using default installation path)
export PATH=$PATH:/home/${USER}/miniconda3/bin/conda

#4. Disable default activation
/home/${USER}/miniconda3/bin/conda config --set auto_activate_base false

#5. Restart terminal / Open new Shell then check conda command availability
conda --version

#6. (Optional) Setup git. Reference: https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup

sudo apt-get install -y python3 python3-pip

#7. installing aws-cli
sudo apt-get install -y unzip 
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install


# if Ubuntu linux
#sudo apt-get update && sudo apt-get -y upgrade
#sudo apt-get install -y docker.io
#sudo service docker start
#sudo usermod -a -G docker ubuntu
#sudo docker pull casualcodersm/iot_device_simulator
#sudo docker run -it casualcodersm/iot_device_simulator
#Follow steps specified in ReadMe.md


# if Amazon Linux2
#sudo yum update && sudo yum -y upgrade
#sudo yum install -y docker
#sudo service docker start
#sudo usermod -a -G docker ec2-user
#sudo systemctl enable docker.service
#sudo systemctl start docker.service
#sudo docker pull casualcodersm/iot_device_simulator
#sudo docker run -it casualcodersm/iot_device_simulator
#Follow steps specified in ReadMe.md