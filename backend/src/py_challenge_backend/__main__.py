import os
from argparse import ArgumentParser

import uvicorn

parser = ArgumentParser(
    "py_challenge backend",
    description="A REST API application to run as the backend required to complete the py_challenge proposed at https://github.com/jfaldanam/py_challenge",
)
subparsers = parser.add_subparsers(dest="command")

run_parser = subparsers.add_parser("run", help="Run the application")
run_parser.add_argument(
    "--host",
    type=str,
    default="0.0.0.0",
    help="The network address to listen to. Default is: '0.0.0.0'",
)
run_parser.add_argument(
    "--port",
    type=int,
    default=8778,
    help="The network port to listen to. Default is: 8778",
)
run_parser.add_argument(
    "--reload",
    action="store_true",
    help="Enable auto-reload of the server when the code changes",
)
run_parser.add_argument(
    "--data-service",
    type=str,
    default="http://localhost:8777",
    help="The URL of the data service to retrieve the challenge data from",
)
run_parser.add_argument(
    "--minio-service",
    type=str,
    default="localhost:9005",
    help="The URL of the MinIO service to store the trained models. (Without the protocol)",
)
run_parser.add_argument(
    "--minio-console",
    type=str,
    default="http://localhost:9006",
    help="The URL of the MinIO console.",
)
run_parser.add_argument(
    "--minio-access-key",
    type=str,
    default="minioadmin",
    help="The access key to connect to the MinIO service",
)
run_parser.add_argument(
    "--minio-secret-key",
    type=str,
    default="minioadmin",
    help="The secret key to connect to the MinIO service",
)
run_parser.add_argument(
    "--minio-bucket",
    type=str,
    default="py-challenge",
    help="The MinIO bucket to store the trained models",
)

args = parser.parse_args()

match args.command:
    case "run":
        # The utils models request the data service URL from the environment variable
        os.environ["PY_CHALLENGE_DATA_SERVICE"] = args.data_service
        os.environ["PY_CHALLENGE_MINIO_SERVICE"] = args.minio_service
        os.environ["PY_CHALLENGE_MINIO_CONSOLE"] = args.minio_console
        os.environ["PY_CHALLENGE_MINIO_ACCESS_KEY"] = args.minio_access_key
        os.environ["PY_CHALLENGE_MINIO_SECRET_KEY"] = args.minio_secret_key
        os.environ["PY_CHALLENGE_MINIO_BUCKET"] = args.minio_bucket

        # Run the application
        uvicorn.run(
            "py_challenge_backend.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    case _:
        parser.print_help()
        exit(1)
