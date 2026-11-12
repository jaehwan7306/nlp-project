"""A short explicit CLI; heavyweight imports occur only inside model commands."""

import argparse
import json
import sys
import traceback

from . import CONDITIONS
from .data import audit_data, fetch_data, prepare_data
from .experiment import compress, diagnose, freeze, initialize, inspect_inputs, judge
from .metrics import analyze
from .prompts import prompt_spec
from .runtime import download_models, lock_models
from .settings import load_config
from .storage import read_json, write_json


def parser():
    cli = argparse.ArgumentParser(description="ContractNLI main experiment: Original + 4 question-free compression conditions.")
    cli.add_argument("--debug", action="store_true", help="show full traceback on failure")
    commands = cli.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-data", help="download official raw JSON files without overwriting")
    fetch.add_argument("--dest", default="data")
    audit = commands.add_parser("audit-data", help="validate all splits and make duplicate review candidates")
    audit.add_argument("--data", default="data")
    audit.add_argument("--out", default="data_audit")
    audit.add_argument("--threshold", type=float, default=0.95)
    prepare = commands.add_parser("prepare", help="apply human duplicate reviews and freeze prepared data")
    prepare.add_argument("--audit", default="data_audit/audit.json")
    prepare.add_argument("--review", default="data_audit/review.csv")
    prepare.add_argument("--out", default="prepared")
    lock = commands.add_parser("lock-models", help="resolve and store immutable model revision SHAs")
    lock.add_argument("--config", default="config.toml")
    lock.add_argument("--out", default="model_lock.json")
    download = commands.add_parser("download-models", help="download pinned model weights to the HF cache")
    download.add_argument("--config", default="config.toml")
    download.add_argument("--models", default="model_lock.json")
    inspect = commands.add_parser("inspect-inputs", help="tokenizer-only context length and span checks")
    inspect.add_argument("--config", default="config.toml")
    inspect.add_argument("--prepared", default="prepared")
    inspect.add_argument("--models", default="model_lock.json")
    inspect.add_argument("--out", default="runs/input_stats")
    init = commands.add_parser("init-run", help="create immutable run snapshots; Test requires frozen protocol")
    init.add_argument("--config", default="config.toml")
    init.add_argument("--prepared", default="prepared")
    init.add_argument("--models", default="model_lock.json")
    init.add_argument("--split", choices=("dev", "test"), required=True)
    init.add_argument("--out", required=True)
    init.add_argument("--documents", type=int, help="Dev preflight only: select this many longest source documents")
    init.add_argument("--protocol", help="frozen protocol directory, required for Test")
    for name in ("compress", "judge", "diagnose", "analyze"):
        command = commands.add_parser(name)
        command.add_argument("--run", required=True)
        if name in ("compress", "judge"):
            command.add_argument("--resume", action="store_true", help="skip completed rows and recover an interrupted final journal line")
        if name in ("judge", "analyze"):
            command.add_argument("--conditions", nargs="+", choices=CONDITIONS, help="default: all five; original alone permits baseline-first evaluation")
    frozen = commands.add_parser("freeze", help="freeze completed full Dev technical protocol, without selecting on accuracy")
    frozen.add_argument("--run", required=True)
    frozen.add_argument("--out", default="protocol")
    prompts = commands.add_parser("show-prompts", help="print/save every exact instruction and insertion format; no models")
    prompts.add_argument("--out")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.version_info[:2] != (3, 11):
        print("ERROR: Use Python 3.11 for this reproducible experiment.", file=sys.stderr)
        return 2
    try:
        match args.command:
            case "fetch-data":
                fetch_data(args.dest)
            case "audit-data":
                audit_data(args.data, args.out, args.threshold)
            case "prepare":
                prepare_data(args.audit, args.review, args.out)
            case "lock-models":
                lock_models(load_config(args.config), args.out)
            case "download-models":
                download_models(load_config(args.config), read_json(args.models))
            case "inspect-inputs":
                inspect_inputs(args.config, args.prepared, args.models, args.out)
            case "init-run":
                initialize(args.config, args.prepared, args.models, args.split, args.out, args.documents, args.protocol)
            case "compress":
                compress(args.run, args.resume)
            case "judge":
                judge(args.run, args.conditions, args.resume)
            case "diagnose":
                diagnose(args.run)
            case "analyze":
                analyze(args.run, args.conditions)
            case "freeze":
                freeze(args.run, args.out)
            case "show-prompts":
                if args.out:
                    write_json(args.out, prompt_spec())
                else:
                    print(json.dumps(prompt_spec(), ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        print("\nInterrupted. Completed records are retained; resume the same stage with --resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.debug:
            traceback.print_exc()
        else:
            print(f"ERROR: {exc}\nUse --debug before the command for a full traceback.", file=sys.stderr)
        return 1
    return 0
