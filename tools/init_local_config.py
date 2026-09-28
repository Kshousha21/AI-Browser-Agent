"""Create a private local configuration once, without displaying credentials."""
from __future__ import annotations

import argparse
from pathlib import Path
import secrets

ROOT = Path(__file__).absolute().parents[1]
TOKEN_PLACEHOLDER = "BROWSER_AUTH__TOKEN="


def initialize(template: Path, destination: Path) -> bool:
    """Return True when created; never read or replace an existing configuration."""
    if destination.exists() or destination.is_symlink():
        return False
    lines = template.read_text(encoding="utf-8").splitlines()
    if lines.count(TOKEN_PLACEHOLDER) != 1:
        raise ValueError("The example must contain exactly one blank connection-token setting.")
    lines = ["BROWSER_PROVIDER__ACTIVE=local_agent" if line == "BROWSER_PROVIDER__ACTIVE=" else line
             for line in lines]
    token = secrets.token_urlsafe(32)
    lines[lines.index(TOKEN_PLACEHOLDER)] = TOKEN_PLACEHOLDER + token
    try:
        # Exclusive creation also prevents an overwrite if another process wins the race.
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(lines) + "\n")
    except FileExistsError:
        return False
    return True


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    created = initialize(ROOT / "backend/.env.example", ROOT / "backend/.env")
    print("Created backend/.env with a random connection token."
          if created else "Existing backend/.env preserved; no values were read or changed.")
    print("Open it privately to copy the token into the extension Settings. Never commit it.")


if __name__ == "__main__":
    main()
