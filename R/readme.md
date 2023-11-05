## DKF Implementations | R

> Runs with R 4.3.2 and the packages contained in `renv.lock`.

```
R
install.packages("renv")
renv::restore()
source("exampleUsage.R")
```

Alternatively, with docker:

```
docker build -t dkf-env-r .
docker run --rm -ti -v $(pwd):/home/felixity/src dkf-env-r Rscript exampleUsage.R
```
