## DKF Implementations | Python

> Runs with Python 3.11.6 and requirements as outlined in `requirements.txt`.

For example,

```
python3 -m venv turquoise
source turquoise/bin/activate
pip3 install -r requirements.txt
python3 exampleUsage.py
```

Alternatively, with docker:

```
docker build -t dkf-env-python .
docker run --rm -ti -v $(pwd):/home/felixity/src dkf-env-python python exampleUsage.py
```
