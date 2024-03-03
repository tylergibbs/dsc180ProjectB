FROM ubuntu:20.04
RUN apt-get update && apt-get install -y python3.11.7 python3.11.7-dev
COPY . .
RUN pip install -r requirements.txt
