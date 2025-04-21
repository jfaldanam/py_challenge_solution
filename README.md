# Structure of this solution

> [!NOTE]
> This repository includes git submodules. To clone it, use the following command: `git submodule update --init --recursive`. This will clone the repository and all its submodules. If you have already cloned the repository, you can update the submodules with the same command.

This solution is structured as follows:
- The solutions to steps 0 and 1 are not covered, they are general guidelines to motivate the the use of Git, Python best practices, and Docker. For 1 you can check the [data-service/](https://github.com/jfaldanam/py_challenge/tree/master/data-service) folder in the repostory describing the challenge.
- The solution to step 2 and 3 is in the [backend/](backend/) folder, check the [README.md](backend/README.md) for more information.
- The solution to step 4 is in the [frontend/](frontend/) folder, check the [README.md](frontend/README.md) for more information.
- The solution to step 5 is split in each folder (Dockerfile's and README.md's) and the final deployment is in the [docker-compose.yaml](docker-compose.yaml) file, instruction on how to run it below.

# How to run the complete solution
To run the complete solution, you need to have Docker and Docker Compose installed. Then, you can run the following command in the root of the repository:


> [!WARNING]
> Please note that this solutions is for educational purposes only. Everything is deployed with default credentials or no authentication and no security measures.
> Putting any database with default credentials open to any public network is a security risk.

```bash
$ docker compose up --build
```
This will build and run the data, backend and frontend services, as well as the MinIO service for S3-like storage.

The data service will be available at http://localhost:8777, the backend will be available at http://localhost:8778, the frontend at http://localhost:8779 and minio at http://localhost:9006 with the default credentials `minioadmin:minioadmin`.

The database contents are stored in volumes under the [`db/`](db/) folder.
