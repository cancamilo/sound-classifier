# Sound classification with machine learning



## Data

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

### Tensorflow installation on mac

https://stackoverflow.com/questions/72964800/what-is-the-proper-way-to-install-tensorflow-on-apple-m1-in-2022

## Notebook implementation

## Model training

## Running the service

## Deploying to AWS Elastic Beanstalk

### Configure Linux instance on AWS

- Install anaconda 

`wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh`

accept terms and condition, accept defaults

- Install docker

`sudo apt update`
`sudo apt install docker.io

- Install docker compose

`wget https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-x86_64 -O docker-compose`

go to their github to find recent versions

Also modify the path:

nanon .bashrc

and write  in that file:

export PATH="${HOME}/soft:{PATH}"

execute `source .bashrc` and check `which docker-compose`\

to enable executing docker without sudo:

`sudo usermod -aG docker $USER`

logout and ssh to the server again and it should work.








