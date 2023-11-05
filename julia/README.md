## DKF Implementations | Julia

> Runs with Julia 1.9.3 and packages as given in the Project.toml file.

```
julia --project=. -e 'import Pkg; Pkg.instantiate()'
julia --project=. exampleUsage.jl
```

Alternatively, with docker:

```
docker build -t dkf-env-julia .
docker run --rm -ti -v $(pwd):/home/felixity/src dkf-env-julia julia --project=/opt/venv/ exampleUsage.jl
```
