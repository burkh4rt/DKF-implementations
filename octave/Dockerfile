# use this file with:
# docker build -t dkf-env-octave .
# docker run --rm -ti -v $(pwd):/home/felixity/src dkf-env-octave octave exampleUsage.m

FROM gnuoctave/octave:6.2.0

RUN useradd felixity

ADD --chown=felixity \
  https://raw.githubusercontent.com/burkh4rt/DKF-implementations/master/data/z.csv \
  https://raw.githubusercontent.com/burkh4rt/DKF-implementations/master/data/x.csv \
  /home/felixity/data/

# switch to non-root user
USER felixity
WORKDIR /home/felixity/src