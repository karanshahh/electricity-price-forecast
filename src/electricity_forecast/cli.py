"""CLI entry point."""

import argparse


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(prog="electricity-forecast")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Placeholder subcommands
    subparsers.add_parser("fetch", help="Fetch raw data")
    subparsers.add_parser("features", help="Build features")
    subparsers.add_parser("train", help="Train model")
    subparsers.add_parser("backtest", help="Run backtest")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
    else:
        print(f"Running: {args.command} (placeholder)")


if __name__ == "__main__":
    main()
