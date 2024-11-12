
FROM python:3.10-slim
WORKDIR /app

RUN apt-get update
RUN apt-get -yy install bart

COPY scripts/ /app/

RUN pip install numpy scipy matplotlib mrd-python

CMD ["python3"]
