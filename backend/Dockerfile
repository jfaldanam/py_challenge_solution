FROM python:3.11-slim

WORKDIR /code/py_challenge_backend/

COPY pyproject.toml README.md /code/py_challenge_backend/
COPY src/ /code/py_challenge_backend/src/

RUN python -m pip install .

ENTRYPOINT [ "python", "-m", "py_challenge_backend", "run", "--port", "8778" ]