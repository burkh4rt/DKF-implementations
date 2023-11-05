# use this file with:
# docker build -t dkf-env-r .
# docker run --rm -ti -v $(pwd):/home/felixity/src dkf-env-r Rscript exampleUsage.R
#
# to troubleshoot:
# docker run --rm -ti -v $(pwd):/home/felixity/src --entrypoint /bin/bash dkf-env-r

FROM r-base:4.3.2

# install requirements
COPY renv.lock ./
RUN  R -e "install.packages('remotes', repos = c(CRAN = 'https://cloud.r-project.org'))" \
 && R -e "remotes::install_github('rstudio/renv')" \
 && R -e "renv::restore()" \
 && useradd felixity

ADD --chown=felixity \
  https://raw.githubusercontent.com/burkh4rt/DKF-implementations/master/data/z.csv \
  https://raw.githubusercontent.com/burkh4rt/DKF-implementations/master/data/x.csv \
  /home/felixity/data/

# switch to non-root user
USER felixity
WORKDIR /home/felixity/src
