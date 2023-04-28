FROM casualcodersm/phm-datasets as datasets
FROM ubuntu as main

############################################################
# install dependent dev tools
############################################################
RUN apt-get update -y && apt-get -y upgrade
RUN apt-get install -y wget curl tar unzip p7zip-full unrar nano vim sudo procps
RUN apt-get install -y python3-pip gcc default-jdk

############################################################
# create user as per AWS documentation for greengrass
############################################################
RUN useradd --system --create-home gcc_user
RUN groupadd --system gcc_group

############################################################
#creating user with sudo permission
############################################################
#RUN useradd -rm -d /home/sm sm && echo "sm:bvadproject" | chpasswd && adduser sm sudo
#RUN adduser sudo
#USER sm
#WORKDIR /home/sm

############################################################
# Copy the IMS Dataset
############################################################
COPY --from=datasets /root/datasets /root/datasets
WORKDIR /root

############################################################
# Install and Configur Conda (Miniconda) Environment
############################################################
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# Crete the conda environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new enviroment
SHELL ["conda", "run", "-n", "cds", "/bin/bash", "-c"]

#Demonstrate the environment is activated
RUN echo "Make sure tensorflow is installed:"
RUN python -c "import tensorflow"

############################################################
# Setup Environment for Mimic IOT device
############################################################
#Example code to run when container is started
#COPY test_run.py .
#ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "cds", "python", "test_run.py"]

############################################################
# Device specific environment variables.
# Default values set here for documentation. 
# Actual values to be set while running container with -e param
# 
############################################################
ENV device_name="Mimic-BearingVibrationSensor"
ENV predict_for_dataset=1
