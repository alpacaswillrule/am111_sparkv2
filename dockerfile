FROM amazoncorretto:8

RUN yum -y update
RUN yum -y install yum-utils
RUN yum -y groupinstall development

RUN yum list python3*
RUN yum -y install python3 python3-dev python3-pip python3-virtualenv

RUN python -V
RUN python3 -V

ENV PYSPARK_DRIVER_PYTHON python3
ENV PYSPARK_PYTHON python3
ENV PYTHONFILETORUN ./training_fin_classfier.py
ENV NUMRECORDS 400
ENV NUMWARCS 15
ENV RANSAMPLE N

COPY requirements.txt ./
COPY articlespar.parquet ./articlespar.parquet
COPY training_fin_classfier.py ./
COPY sentdat ./sentdat
COPY models ./models
COPY sentiment_run.py ./
COPY sentiment_long_optim.py ./

RUN pip3 install --upgrade pip
RUN pip3 install -r ./requirements.txt

CMD [ "sh", "-c", "python3 $PYTHONFILETORUN" ]