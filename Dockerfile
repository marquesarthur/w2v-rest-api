# pull base image
FROM debian:jessie

FROM msarthur/semafor-base:latest

VOLUME /var/log/docker

COPY . /tmp/src

WORKDIR /tmp/src/

RUN apk --update add supervisor curl python uwsgi \
    bash nginx python-dev py-pip

RUN pip install -r requirements.txt

ENTRYPOINT [ "python" ]

EXPOSE 5001

CMD ["web_app.py"]