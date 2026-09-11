"""D-04: `python -m data.validate <output-prefix> --profile <path>` — the
argv-only shim over `data.validation.validate.validate_dataset`. Zero
validation logic lives here; this module parses argv, loads the profile,
calls `validate_dataset`, prints the report path, and returns the outcome's
exit code — following `data/tools/inspect_mcap.py`'s
`build_parser`/`main(argv)`/`__main__` shape.
"""

import argparse
import sys
from pathlib import Path

from data.contracts.profile import load_profile
from data.validation.report import VALIDATION_REPORT_FILENAME
from data.validation.validate import validate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data.validate",
        description=(
            "Runs the post-write validation check registry against an "
            "already-written LeRobot v2.1 dataset prefix. Reports and "
            "amends provenance only — never repairs, trims, or re-encodes "
            "anything on disk."
        ),
    )
    parser.add_argument("out_dir", metavar="output-prefix")
    parser.add_argument("--profile", required=True, metavar="profile-path")
    parser.add_argument(
        "--use-local-golden",
        action="store_true",
        help=(
            "Opt in to resolving the local, gitignored data/goldens/baseline.json "
            "pointer for the regression.golden_fingerprint (REGR-01) check. "
            "Without this flag, that local pointer is ignored entirely and the "
            "check reports unevaluated -- a real baseline.json left on disk from "
            "a prior local golden-build session cannot affect this run's verdict "
            "unless this flag is passed."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    profile = load_profile(args.profile)
    outcome = validate_dataset(args.out_dir, profile, use_local_golden=args.use_local_golden)

    report_path = Path(args.out_dir) / "meta" / "data_engine" / VALIDATION_REPORT_FILENAME
    print(str(report_path))

    return outcome.exit_code


if __name__ == "__main__":
    sys.exit(main())
