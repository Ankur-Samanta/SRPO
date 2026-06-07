#!/usr/bin/env python3
"""Merge sharded evaluation output JSONs into a single file."""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge eval shards")
    parser.add_argument("--inputs", nargs="+", required=True, help="Shard JSON files")
    parser.add_argument("--output", required=True, help="Merged output JSON")
    args = parser.parse_args()

    merged_results = []
    config = None

    for path in sorted(args.inputs):
        with open(path) as f:
            shard = json.load(f)
        if config is None:
            config = shard["config"]
        merged_results.extend(shard["results"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({"config": config, "results": merged_results}, f, indent=2)

    print(f"Merged {len(args.inputs)} shards -> {len(merged_results)} problems -> {output_path}")


if __name__ == "__main__":
    main()
