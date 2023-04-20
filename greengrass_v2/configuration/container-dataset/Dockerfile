FROM ubuntu

############################################################
# install dependent dev tools
############################################################
RUN apt-get update -y && apt-get -y upgrade
RUN apt-get install -y wget curl tar unzip p7zip-full unrar nano vim sudo procps

WORKDIR /root/datasets
ADD https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip phm_ims_bearing_dataset.zip

RUN unzip phm_ims_bearing_dataset.zip && rm phm_ims_bearing_dataset.zip && mv '4. Bearings' 'phm-ims-datasets'
WORKDIR /root/datasets/phm-ims-datasets
RUN 7z x -y IMS.7z && rm IMS.7z && unrar x -y 1st_test.rar && unrar x -y 2nd_test.rar && unrar e -y 3rd_test.rar 3rd_test/ && rm *.rar
WORKDIR /root