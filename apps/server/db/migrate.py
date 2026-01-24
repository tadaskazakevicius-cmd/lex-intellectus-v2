from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import re

MIGRATION_RE = re.compile(r"^(?P<ver>\d{4})_.*\.sql$")


@dataclass(frozen=True)
class Migration:
    version: int
    path: Path


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON;")
    return con


def get_user_version(con: sqlite3.Connection) -> int:
    return int(con.execute("PRAGMA user_version;").fetchone()[0])


def load_migrations(migrations_dir: Path) -> list[Migration]:
    migrations: list[Migration] = []
    if not migrations_dir.exists():
        return migrations

    for p in migrations_dir.iterdir():
        if not p.is_file():
            continue
        m = MIGRATION_RE.match(p.name)
        if not m:
            continue
        migrations.append(Migration(version=int(m.group("ver")), path=p))

    migrations.sort(key=lambda x: x.version)
    return migrations


def apply_sql_file(con: sqlite3.Connection, path: Path) -> None:
    sql = path.read_text(encoding="utf-8")
    con.executescript(sql)


def ensure_base_schema(con: sqlite3.Connection, schema_sql_path: Path) -> None:
    if schema_sql_path.exists():
        con.executescript(schema_sql_path.read_text(encoding="utf-8"))


def migrate(db_path: Path, schema_sql_path: Path, migrations_dir: Path) -> int:
    con = connect(db_path)
    try:
        ensure_base_schema(con, schema_sql_path)

        current = get_user_version(con)
        migrations = load_migrations(migrations_dir)
        to_apply = [m for m in migrations if m.version > current]

        if not to_apply:
            return current

        for mig in to_apply:
            with con:
                apply_sql_file(con, mig.path)

        return get_user_version(con)
    finally:
        con.close()


if __name__ == "__main__":
    root = Path.cwd()
    db_path = root / ".localdata" / "app.db"

    schema_sql = Path("apps/server/db/schema.sql")
    migrations_dir = Path("apps/server/db/migrations")

    v = migrate(db_path, schema_sql, migrations_dir)
    print(f"DB migrated. user_version={v}, db={db_path}")
