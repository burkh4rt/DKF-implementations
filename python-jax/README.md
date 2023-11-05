## DKF Implementations | Jax

> Runs with Python 3.11.6 with Jax 0.4.20 and other requirements as outlined in
> `requirements.txt`.

For example,

```
python3 -m venv turquoise
source turquoise/bin/activate
pip3 install -r requirements.txt
python3 exampleUsage.py
```

Alternatively, with docker:

```
docker build -t dkf-env-jax .
docker run --rm -ti -v $(pwd):/home/felixity/src dkf-env-jax python exampleUsage.py
```
