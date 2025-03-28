# How the frontend is implemented?
The frontend is implemented using streamlit to implement a web interface.

# Deployment
First, the backend described at [../backend/](../backend/) must be deployed.

After deploying the backend, for instructions on how to deploy the frontend run:

```bash
# Install the package including the development dependencies
$ pip install -e '.[dev]'
# Run the frontend
$ python -m py_challenge_frontend run --help
```

The frontend uses sqlite3 to store the data, by default is stored in the file `py_challenge.sqlite`.

## Deployment on Docker

To build the image run:
```bash
$ docker build -t py_challenge_frontend:latest .
```

Then to run the API use:
```bash
$ docker run -v ./db:/data -p 8779:8779 py_challenge_frontend:latest
```

Please note the use of a volume to persist the data.
