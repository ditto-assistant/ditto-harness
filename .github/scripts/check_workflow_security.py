#!/usr/bin/env python3
"""Fail CI when workflow changes weaken the repository's trust boundary."""

import re
import sys
from pathlib import Path

WORKFLOW_ROOTS = (Path(".github/workflows"),)
PRIVILEGED_TRIGGERS = {
    "pull_request_target",
    "workflow_run",
    "issue_comment",
    "repository_dispatch",
}
TRIGGER_RE = re.compile(r"^\s*(" + "|".join(sorted(PRIVILEGED_TRIGGERS)) + r")\s*:")
USES_RE = re.compile(r"\buses:\s*([^\s#]+)")
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def main() -> int:
    failures = []
    for root in WORKFLOW_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_file() or path.suffix not in {".yml", ".yaml", ".disabled"}:
                continue
            for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
                line = raw_line.split("#", 1)[0]
                trigger = TRIGGER_RE.match(line)
                if trigger:
                    failures.append(
                        f"{path}:{line_number}: privileged trigger "
                        f"{trigger.group(1)!r} is forbidden"
                    )
                action = USES_RE.search(line)
                if action:
                    target = action.group(1)
                    if not target.startswith(("./", "docker://")):
                        _, separator, revision = target.rpartition("@")
                        if not separator or not FULL_SHA_RE.fullmatch(revision):
                            failures.append(
                                f"{path}:{line_number}: external action must use a full "
                                f"40-character commit SHA: {target}"
                            )
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("workflow security policy passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
