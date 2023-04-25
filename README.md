# Anomaly detection in Bearing Vibration Measurements

Academic Project for Automation of Condition Based Monitoring (CBM). Project uses IMS dataset to detect anomaly of Bearing part using Vibration Data

Computation Data Science (CDS) - Cohort 4 - Group 7 members:
1. AnushKumar Jawahar – Main Author of Autoencoder-LSTM
2. Debmalya Chandra – Team meeting scheduler, Author of Proposal document preparation
3. Priyashree Dey – Co-Author of Streamlit app (IoT deployment part)
4. Ram Manohar Rai – Author of Model “Autoencoder-LSTM-Frequency Domain”
5. Shailesh Marathe – Team coordinator & Author of IoT based deployment & Co-Author of Autoencoder LSTM
6. Sridharan Ramar – Team Captain & Author of Supervised models & Author of Streamlit based Dashboard deployment.
7. Suman Banerjee – Architect, Co-Author of Presentation preparation.
8. Tejaswini Indu – Preliminary Explorer of Project Topic & Co-Author of Proposal document presentation. 


## Overview and Problem Statement: 
Bearings play an important role in rotational machines as their failure directly leads to breakdown in the machinery. Its condition impacts the operational performance, service life and efficiency of the machines.

**Domain:** Manufacturing/Engineering

**Techniques:** Anomaly detection

**Motivation:** Anomaly detection or outlier detection is the identification of rare items, events or observations that differ significantly from the majority of the data. Therefore, identifying bearing fault is of utmost importance.

**Goal of Project:**
    - In this study, anomaly detection has to be used as a prognostic tool. The goal of this study is to create a prognostic that predicts the anomaly in vibration of machine bearings, using machine learning approaches.
  - Model is build using the data the signals data measured over the lifetime of the bearings until failure.
  - An automated predictive warning system can be designed to help in avoiding a gear failure and flagging the machine for maintenance or repair.

## Data description: 
- The IMS Bearing Dataset consists of data from three sets of experiments, each containing measurements for 4 bearings. 
  - The bearings were "run to failure" under constant load and running conditions. 
    - The vibration measurement signals are provided for the datasets over the lifetime of the bearings until failure.
    - Failure occurred after 100 million cycles with a crack in the outer race. 
  - As the equipment was run until failure, 
    - Data from the first two days of operation was used as training data to represent normal and “healthy” equipment.
    - The remaining part of the datasets for the time leading up to the bearing failure was used as test data, to evaluate whether the different methods could detect the bearing degradation in advance of the failure.
  - Each data set consists of individual files that are 1-second vibration signal snapshots recorded at specific intervals.
  - Each file consists of 20,480 points with the sampling rate set at 20 kHz. 
  - The file name indicates when the data was collected. Each record (row) in the data file is a data point. Larger intervals of time stamps (shown in file names) indicate resumption of the experiment on the next working day.

## Applications:
Prediction and flagging of upcoming bearing malfunctions in advance of the actual failure

## Data-References:
**Original :** 
1. [NASA Prognostics Center of Excellence Data Set Repository | Mirror by Phm (recommended)](https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip)
2. [Kaggle NASA Bearing Dataset (Recommended)](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset)
3. [Nasa 4. Bearings (seems broken- NOT recommended)](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

**Associated Papers:**
1. [Analysis of the Rolling Element Bearing data set of the Center for Intelligent Maintenance Systems of the University of Cincinnati](https://hal.science/hal-01715193)
2. [Wavelet filter-based weak signature detection method and its application on rolling element bearing prognostics](https://www.researchgate.net/publication/223556476_Wavelet_filter-based_weak_signature_detection_method_and_its_application_on_rolling_element_bearing_prognostics)

**Additional References:**
1. [Deep Learning Algorithms for Bearing Fault Diagnostics– A Comprehensive Review by Shen Zhang & all](https://arxiv.org/pdf/1901.08247)
  
2. [AnAutomated Auto-encoder Correlation-based Health-Monitoring and Prognostic Method for Machine Bearings by Ramin M. Hasani & all](https://arxiv.org/pdf/1703.06272.pdf)
  


