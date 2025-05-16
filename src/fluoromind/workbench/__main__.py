"""Main entry point for running the FluoroMind interface."""

import argparse
import os
import logging
from .simpleui import simple_ui
from .config import get_workdir, update_workdir, default_workdir
from .. import __version__


def setup_logging(debug: bool) -> None:
    """Configure logging settings."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m fluoromind.workbench",
        description="UI for FluoroMind - Advanced Fluorescence Analysis Library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Server configuration group
    server_group = parser.add_argument_group("Server Configuration")
    server_group.add_argument("--port", type=int, help="Port to run the server on (default: 7860)", default=7860)
    server_group.add_argument(
        "--host", type=str, help="Host address to bind to (default: 127.0.0.1)", default="127.0.0.1"
    )
    server_group.add_argument(
        "--share",
        action="store_true",
        help="Enable sharing via a public URL for remote access (requires internet connection)",
    )

    # Directory configuration group
    dir_group = parser.add_argument_group("Directory Configuration")
    dir_group.add_argument(
        "--root-dir",
        type=str,
        help="Root directory for file exploration (default: current working directory)",
        default=os.getcwd(),
    )
    dir_group.add_argument(
        "--working-dir",
        type=str,
        help="Directory for temporary files (default: system temp directory)",
        default=None,
    )

    # Application configuration group
    app_group = parser.add_argument_group("Application Configuration")
    app_group.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")
    app_group.add_argument("--version", action="version", version=f"FluoroMind v{__version__}")

    args = parser.parse_args()

    # Validate root directory
    if not os.path.isdir(args.root_dir):
        parser.error(f"Root directory does not exist: {args.root_dir}")

    return args


def display_banner(args: argparse.Namespace) -> None:
    """Display application banner with configuration details."""
    banner = f"""FluoroMind v{__version__}

Configuration:
- Server URL: http://{args.host}:{args.port}
- Network Access: {'Enabled' if args.share else 'Disabled'}
- Debug Mode: {'Enabled' if args.debug else 'Disabled'}
- Root Directory: {args.root_dir}
- Working Directory: {get_workdir()}
"""
    print(banner)


def main() -> None:
    """Main entry point for the FluoroMind interface."""
    args = parse_args()
    setup_logging(args.debug)

    if args.working_dir:
        update_workdir(args.working_dir)
    else:
        update_workdir(default_workdir())

    display_banner(args)
    logging.info("Starting FluoroMind server...")

    try:
        simple_ui.launch(
            server_name=args.host,
            server_port=args.port,
            debug=args.debug,
            root_path=args.root_dir,
            share=args.share,
        )
    except KeyboardInterrupt:
        logging.info("Shutting down server...")
    except Exception as e:
        logging.error(f"Error starting server: {str(e)}")
        raise


if __name__ == "__main__":
    main()
