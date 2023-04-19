FROM ubuntu

############################################################
# install dependent dev tools
############################################################
RUN apt-get update -y && apt-get -y upgrade
RUN apt-get install -y wget curl tar unzip p7zip-full unrar nano vim sudo procps
RUN apt-get install -y python3-pip gcc default-jdk

############################################################
# create user as per AWS documentation
############################################################
RUN useradd --system --create-home gcc_user
RUN groupadd --system gcc_group

############################################################
#creating user with sudo permission
############################################################
#RUN useradd -rm -d /home/sm sm && echo "sm:sm" | chpasswd && adduser sm sudo
#RUN adduser sudo
#USER sm
#WORKDIR /home/sm

############################################################
#
############################################################

############################################################
# copy data
############################################################
WORKDIR /root/datasets
ADD https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip phm_ims_bearing_dataset.zip

RUN unzip phm_ims_bearing_dataset.zip && rm phm_ims_bearing_dataset.zip && mv '4. Bearings' 'phm-ims-datasets'
WORKDIR /root/datasets/phm-ims-datasets
RUN 7z x -y IMS.7z && rm IMS.7z && unrar x -y 1st_test.rar && unrar x -y 2nd_test.rar && unrar e -y 3rd_test.rar 3rd_test/ && rm *.rar
WORKDIR /root

ENV select_data_set=1
ENV select_bearing=1
ENV select_input_stepsize=3000

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version
RUN conda install -y tensorflow
