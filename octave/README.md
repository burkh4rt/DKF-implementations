## DKF Implementations | Octave (compatable with Matlab)

> Runs with Octave 6.2.0.

For example,

```
octave exampleUsage.m
```

Alternatively, with docker:

```
docker build -t dkf-env-octave .
docker run --rm -ti -v $(pwd):/home/felixity/src dkf-env-octave octave exampleUsage.m
```
