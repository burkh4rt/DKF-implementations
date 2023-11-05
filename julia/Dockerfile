# use this file with:
# docker build -t dkf-env-julia .
# docker run --rm -ti -v $(pwd):/home/felixity/src dkf-env-julia julia --project=/opt/venv/ exampleUsage.jl
#
# to troubleshoot:
# docker run --rm -ti -v $(pwd):/home/felixity/src --entrypoint /bin/bash dkf-env-julia

FROM julia:1.9.3-bullseye

# install requirements
COPY Project.toml /opt/venv/
RUN useradd felixity \
 && chown -R felixity /opt/venv/ \
 && mkdir /home/felixity/ \
 && chown -R felixity /home/felixity/
USER felixity
RUN julia --project=/opt/venv/ -e 'import Pkg; Pkg.instantiate()'

ADD --chown=felixity \
  https://raw.githubusercontent.com/burkh4rt/DKF-implementations/master/data/z.csv \
  https://raw.githubusercontent.com/burkh4rt/DKF-implementations/master/data/x.csv \
  /home/felixity/data/

WORKDIR /home/felixity/src