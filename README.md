# Sound classification with machine learning



## Data

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








