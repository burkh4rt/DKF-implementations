[![DOI](https://zenodo.org/badge/264787686.svg)](https://zenodo.org/badge/latestdoi/264787686)

## DKF Implementations

_This repository contains code implementations for the Discriminative Kalman
Filter. [^1][^2][^3][^4][^5]_

We currently offer example code for filtering using the following languages:

- Julia
- Octave (Matlab)
- Python
- R

The data consists of run 1 from the pre-processed Flint data found in
[burkh4rt/Discriminative-Kalman-Filter](https://github.com/burkh4rt/Discriminative-Kalman-Filter).

[^1]:
    M. Burkhart, D. Brandman, B. Franco, L. Hochberg, & M. Harrison. The
    Discriminative Kalman Filter for Bayesian Filtering with Nonlinear and
    Nongaussian Observation Models. Neural Computation 32 (2020)
    [[link](https://doi.org/10.1162/neco_a_01275)]
    [[implementation](https://github.com/burkh4rt/Discriminative-Kalman-Filter)]

[^2]:
    M. Burkhart. “A Discriminative Approach to Bayesian Filtering with
    Applications to Human Neural Decoding.” Ph.D. Dissertation, Brown
    University (2019) [[link](https://doi.org/10.26300/nhfp-xv22)]

[^3]:
    D. Brandman, M. Burkhart, J. Kelemen, B. Franco, M. Harrison, & L.
    Hochberg. Robust Closed-Loop Control of a Cursor in a Person with
    Tetraplegia using Gaussian Process Regression. Neural Computation 30 (2018)
    [[link](https://doi.org/10.1162/neco_a_01129)]

[^4]:
    D. Brandman, T. Hosman, J. Saab, M. Burkhart, B. Shanahan, J. Ciancibello,
    et al. Rapid calibration of an intracortical brain computer interface for
    people with tetraplegia. Journal of Neural Engineering 15 (2018)
    [[link](https://doi.org/10.1088/1741-2552/aa9ee7)]

[^5]:
    M. Burkhart. Discriminative Bayesian filtering lends momentum to the
    stochastic Newton method for minimizing log-convex functions. Optimization
    Letters 17 (2023) [[link](https://doi.org/10.1007/s11590-022-01895-5)]

<!---
format code with:
```
prettier --write --print-width 79 --prose-wrap always **/*.md
black -l 79 python*/
R -e 'styler::style_dir("R/", transformers = styler::tidyverse_style(strict = TRUE))'
julia -e 'using JuliaFormatter; format("julia/")'
```
-->
