# use this file with:
# docker build -t dkf-env-jax .
# docker run --rm -ti -v $(pwd):/home/felixity/src dkf-env-jax python exampleUsage.py
#
# to troubleshoot:
# docker run --rm -ti -v $(pwd):/home/felixity/src --entrypoint /bin/bash dkf-env-jax

FROM python:3.11.6-bullseye

# create venv in the path
RUN python -m venv /opt/venv
ENV PATH /opt/venv/bin:$PATH

# install requirements
COPY requirements.txt ./
RUN pip install --no-cache -r requirements.txt \
 && useradd felixity \
 && chown -R felixity /opt/venv

ADD --chown=felixity \
  https://raw.githubusercontent.com/burkh4rt/DKF-implementations/master/data/z.csv \
  https://raw.githubusercontent.com/burkh4rt/DKF-implementations/master/data/x.csv \
  /home/felixity/data/

# switch to non-root user
USER felixity
WORKDIR /home/felixity/src
