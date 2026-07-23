"""Doctor CLI command — human-readable + JSON config health report."""
from __future__ import annotations

import json
import sys

from model_allocator import config_writer, schema


def cmd_doctor(args) -> int:
    """Run config doctor and report issues."""
    config_dir = None
    if hasattr(args, "config_dir") and args.config_dir:
        config_dir = args.config_dir
    else:
        from model_allocator.cli import _default_config_dir
        config_dir = _default_config_dir()
    raw = config_writer.load_raw(config_dir)
    report = schema.lint_config(raw)

    total_errors = 0
    total_warnings = 0

    for section in ("aliases", "profiles", "roles"):
        for name, issues in report.get(section, {}).items():
            for issue in issues:
                if issue.level == "error":
                    total_errors += 1
                else:
                    total_warnings += 1

    if getattr(args, "json", False):
        # JSON output
        json_report = {}
        for section in ("aliases", "profiles", "roles"):
            json_report[section] = {}
            for name, issues in report.get(section, {}).items():
                json_report[section][name] = [
                    {"level": i.level, "field": i.field, "message": i.message}
                    for i in issues
                ]
        json_report["summary"] = {"errors": total_errors, "warnings": total_warnings}
        print(json.dumps(json_report, indent=2, default=str))
    else:
        # Human-readable output
        if total_errors == 0 and total_warnings == 0:
            print("✅ Config is healthy — no errors or warnings")
        else:
            if total_errors:
                print(f"❌ {total_errors} error(s)")
            if total_warnings:
                print(f"⚠  {total_warnings} warning(s)")
            print()
            for section in ("aliases", "profiles", "roles"):
                for name, issues in report.get(section, {}).items():
                    print(f"[{section}] {name}:")
                    for issue in issues:
                        symbol = "✗" if issue.level == "error" else "⚠"
                        print(f"  {symbol} {issue.field}: {issue.message}")
                    print()

    return 1 if total_errors else 0
