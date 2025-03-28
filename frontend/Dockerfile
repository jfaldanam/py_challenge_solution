FROM python:3.11-slim

WORKDIR /code/py_challenge_frontend/

COPY pyproject.toml README.md /code/py_challenge_frontend/
COPY src/ /code/py_challenge_frontend/src/

RUN python -m pip install .

ENTRYPOINT [ "python", "-m", "py_challenge_frontend", "run", "--port", "8779", "--sqlite-db", "/data/py_challenge.sqlite" ]