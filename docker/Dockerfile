FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y git

# add python requirements
ADD requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

COPY . .
# RUN python setup.py install
# RUN pip install -e .

WORKDIR /
ENTRYPOINT ["/app/docker/entrypoint.sh"]
