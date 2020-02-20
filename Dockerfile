FROM continuumio/miniconda3

# apt dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
		build-essential \
		cmake \
		git \
		curl \
		ca-certificates \
		libjpeg-dev \
		libpng-dev \
		&& rm -rf /var/lib/apt/lists/ 

# python dependencies
RUN python -m pip install setuptools==45.1.0 wheel
RUN conda install pytorch torchvision cpuonly -c pytorch
ADD ./webapp/requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

ADD ./webapp /opt/webapp/
WORKDIR /opt/webapp
# MNIST dataset
RUN python getdata.py

CMD gunicorn --bind 0.0.0.0:$PORT index:server
