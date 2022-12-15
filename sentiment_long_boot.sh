#!/bin/bash
# This script is used to run the sentiment_long_boot.py script, with the arg
#being the ip of master node on emr

scp -i $2 -o StrictHostKeyChecking=accept-new -r "$(pwd)/dockerfile" "$(pwd)/training_fin_classfier.py" "$(pwd)/sentiment_long_optim.py" "$(pwd)/sentdat" "$(pwd)/models" "$(pwd)/requirements.txt" ec2-user@$1:/home/ec2-user

ssh -i $2 ec2-user@$1

mkdir articlespar.parquet

sudo docker build -t sentiment_long_boot .
