from __future__ import annotations

import argparse

from adapter_common import load_canonical_request, write_canonical_response

from .adapter import AlphaBetaCrownAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ivm-alpha-beta-crown",
        description="Phase 1 alpha-beta-CROWN worker scaffold",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the worker input manifest JSON file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the worker result manifest JSON file",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    request = load_canonical_request(args.input)
    adapter = AlphaBetaCrownAdapter()
    response = adapter.run(request)
    write_canonical_response(args.output, response)
    return 0
