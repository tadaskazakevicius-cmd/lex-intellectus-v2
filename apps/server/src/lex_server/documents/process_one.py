from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Process one uploaded document into chunks (D3).")
    p.add_argument("document_id", type=int)
    args = p.parse_args()

    # Support running as module and as script.
    root = Path.cwd()
    src_dir = root / "apps" / "server" / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from lex_server.documents.storage import connect_db
    from lex_server.documents.pipeline import process_document

    con = connect_db()
    try:
        n = process_document(con, args.document_id)
    finally:
        con.close()

    print(n)


if __name__ == "__main__":
    main()

