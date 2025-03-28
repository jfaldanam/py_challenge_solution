import os
import subprocess
import sys
from argparse import ArgumentParser

from py_challenge_frontend import logger


def start_streamlit(
    server_address: str,
    port: int,
    environ: dict[str, str],
) -> subprocess.Popen:
    """
    Start streamlit dashboard.

    We use a subprocess instead of importing the streamlit package
    because it messes with the logging configuration.

    Args:
        data_path: Path to the data directory for Evolver outputs
        resources_path: Path to the resources directory for problem data
        port: Port to run the dashboard on
        logger_level: Set the logging level. Defaults to INFO
        jar: Path to the jar file for Evolver. Defaults to None

    Returns:
        subprocess.Popen: The process running the streamlit dashboard
    """
    # Set up environment variables and arguments
    environment = environ.copy()
    environment["STREAMLIT_BROWSER_SERVER_PORT"] = str(port)
    environment["STREAMLIT_SERVER_PORT"] = str(port)
    environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "FALSE"

    command = [
        sys.executable,  # Run the same python interpreter
        "-m",
        "streamlit",
        "run",
        f"{os.path.dirname(__file__)}/app.py",
        "--server.address",
        server_address,
    ]

    # Start the subprocess with the environment variables
    process = subprocess.Popen(command, env=environment)

    return process


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
    default=8779,
    help="The network port to listen to. Default is: 8779",
)
run_parser.add_argument(
    "--data-service",
    type=str,
    default="http://localhost:8777",
    help="The URL of the data service to retrieve the challenge data from",
)
run_parser.add_argument(
    "--backend-service",
    type=str,
    default="http://localhost:8778",
    help="The URL of the backend service to run predictions on.",
)
run_parser.add_argument(
    "--sqlite-db",
    type=str,
    default="py_challenge.sqlite",
    help="The path to the SQLite database file to use.",
)

args = parser.parse_args()

match args.command:
    case "run":
        # The utils models request the data service URL from the environment variable
        os.environ["PY_CHALLENGE_DATA_SERVICE"] = args.data_service
        os.environ["PY_CHALLENGE_BACKEND_SERVICE"] = args.backend_service
        os.environ["PY_CHALLENGE_SQLITE_DB"] = args.sqlite_db

        # Run the application
        try:
            logger.info("Starting streamlit dashboard")
            streamlit_process = start_streamlit(
                server_address=args.host,
                port=args.port,
                environ=os.environ,
            )
            streamlit_process.wait()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as err:
            logger.fatal(err)
        finally:
            # Make sure to stop the streamlit dashboard to avoid zombie processes
            logger.info("Stopping streamlit dashboard")
            streamlit_process.terminate()
    case _:
        parser.print_help()
        exit(1)
