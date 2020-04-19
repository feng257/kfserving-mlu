FROM tensorflow/tensorflow:1.10.1-devel-py3

WORKDIR /workspace
RUN chmod -R a+w /workspace

COPY tfserver tfserver
COPY kfserving kfserving
COPY third_party third_party

RUN pip install --upgrade pip && pip install -e ./kfserving
RUN pip install -e ./tfserver
ENTRYPOINT ["python", "-m", "tfserver"]