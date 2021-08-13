FROM nvcr.io/nvidia/pytorch:20.10-py3
WORKDIR /workspace
COPY * /workspace/
RUN pip install -r requirements.txt

ARG USER=tom
ARG UID=1001
ARG GID=1001
# default password for user
ARG PW=123qwe
RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | \
      chpasswd
USER ${UID}:${GID}
WORKDIR /home/${USER}
CMD /bin/bash