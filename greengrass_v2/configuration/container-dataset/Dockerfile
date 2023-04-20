FROM ubuntu

############################################################
# install dependent dev tools
############################################################
RUN apt-get update -y && apt-get -y upgrade
RUN apt-get install -y wget curl tar unzip p7zip-full unrar nano vim sudo procps

WORKDIR /root/datasets
ADD https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip phm_ims_bearing_dataset.zip