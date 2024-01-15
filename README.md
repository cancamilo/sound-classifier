# Sound classification with machine learning

This project showcases the usage of machine learning applied to the problem of sound classification. 

Sound classification has a wide range of practical applications across various domains. For example, classification of environmental noise such as traffic noise, construction sounds, or wildlife sounds helps in assessing the impact on urban planning, public health, or wildlife conservation. Also in industrial settings it could help identify malfunctioning machinery or potential safety hazzards. These are only a couple of use cases among thousands that highlight the relevance and versatility of automated sound classification.

This demonstration makes use of a [kaggle dataset](https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50) containing 50 different sound classes from which only 10 classes are selected for simplicity. 

The selected dataset is used to train a convolutional neural network that classifies audio signals into one out of 10 categories. Since the amount of data is limited to only 40 samples per class, this project makes use of data augmentation tecniques in order to have a more robust model. 

For demonstration purposes, the model is then put into a simple streamlit application and depoyed to the cloud to enable easy interaction with end users.

Project outline:

- [Data description](#data-description)
- [Environment setup](#environment-setup)
- [Exploring the data](#exploring-the-data)

## Data Description

The sound data is obtained from [this kaggle dataset](https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50)

The data contains a csv file with the following fields:

filename: reference to the .wav file. type:  object
fold: fold number. type:  int64 
category: label representing the type of sound. type:  object
src_file: file number. type:  int64 

The data contains 40 different sound classes, but for the purpose of this project only the following 10 are selected:

```
{
    'dog': 0,
    'chirping_birds': 1,
    'thunderstorm': 2,
    'keyboard_typing': 3,
    'car_horn': 4,
    'drinking_sipping': 5,
    'rain': 6,
    'breathing': 7,
    'coughing': 8,
    'cat': 9
 }
```

## Environment setup

In order to run the notebooks and scripts provied in this repositoy, you should download [this kaggle dataset](https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50) and save it to the folder **data** in the root of this repository.

For managing the python dependencies and virtual environments I chose conda. The provided [project-dependencies.yml file](project-dependencies.yml) contains the neccesary dependencies to run the notebooks. However note that currently this setting was tried only on Macbook Pro with an M2 chip. If you want to run this in a different system, the tensorflow dependencies have to be changed. As for the rest of the dependencies they can stay the same. With the correct tensorflow dependencies,  the environment can be created as:

```console
conda env create -f project-dependencies.yml
```

Once it is created it can be activated with ```conda activate tf-metal-2```

Also, make sure that you have Docker setup in your system and the aws-cli if you want to deploy the service to the cloud.

References: https://stackoverflow.com/questions/72964800/what-is-the-proper-way-to-install-tensorflow-on-apple-m1-in-2022

### Configure Linux instance on AWS

The previously described environment configuration is used for developemnt purpose. If all you want is to run the streamlit app on a linux virtual machine these are the steps to follow: 

- Create an aws EC2 instance as instructed in [this video](https://www.youtube.com/watch?v=IXSiYkP23zo&ab_channel=DataTalksClub%E2%AC%9B)

- Once the connection to the instance is established we can can install the neccesary packages.

    - Install conda
    ```console
    wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
    ```
    accept terms and condition, accept defaults

    - Install docker

    ```console
    sudo apt update
    sudo apt install docker.io
    ```

    - Install docker compose

    ```console
    wget https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-x86_64 -O docker-compose
    ```
    check docker compose github to find recent versions.

    - Modify the path:

    ```console
    nanon .bashrc
    ```

    and modify the file as follows:

    export PATH="${HOME}/soft:{PATH}"

    execute `source .bashrc` and check `which docker-compose`\

    to enable executing docker without sudo:

    ```console
    sudo usermod -aG docker $USER
    ```

    logout and ssh to the server again and it should work.

- Run the service

First of all this repository should be clone to the linux virtual machine. Given that the previous installation was done succesfully, we can build the docker image and run the container

```console
docker build -t sound-img . 
```

```console
docker run -d -p 8501:8501 sound-img
```

With this, the service should be accesible on the 8501 port. Note that you can set port forwarding on your local machine to make the service accesible from your browser.

## Exploring the data



## Model training

## Running the service

## Deploying to AWS Elastic Beanstalk








