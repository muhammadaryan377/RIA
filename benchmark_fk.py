"""FK inference benchmark (txt Priority 4).

Takes databases with KNOWN (declared) foreign keys, removes those constraints,
re-runs inference on the stripped clone, and compares the inferred relationships
against the original declared set:

    Precision = correct_inferred / total_inferred
    Recall    = correct_inferred / total_declared
    F1        = 2*P*R / (P+R)
    FPR       = wrong_inferred / total_inferred
    FNR       = missed_declared / total_declared

Supported cases (txt table):
  * single-schema databases (northwind, chinook, ...)
  * multi-schema databases (adventureworks, aw_edit, ...) via build_schema_mapping_all
  * messy naming: rename every child FK column to an opaque name (fk_c1, fk_c2, ...)
    so inference must rely on DATA (value overlap + cardinality), not names.

Run:
    python benchmark_fk.py [--dbs northwind adventureworks ...] [--messy]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import schema_agent

ADMIN = "postgresql://{user}:{password}@{host}:{port}/{dbname}".format(
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", "postgres"),
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", "5432"),
    dbname=os.getenv("DB_NAME", "postgres"),
)

DEFAULT_DBS = ["northwind", "chinook", "olist_ecommerce", "retail_fraud", "adventureworks"]

MAX_SOURCE_RATIO = 0.70  # keep at least 30% of the total inferred set


def make_cfg(db):
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "db": db,
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
        "db_type": "postgresql",
        "db_schema": os.getenv("DB_SCHEMA", "public"),
    }


def list_fks(conn, schema=None):
    """Declared FKs, optionally restricted to one schema. Returns
    (table, column, ftable, fcol, schema, fschema) tuples with the
    schema-qualified table names for multi-schema databases.

    Uses pg_constraint (not the information_schema *column_usage* views,
    whose table_schema join silently drops cross-schema foreign keys)."""
    query = """
        SELECT
            refcon.relname AS table_name,
            refatt.attname AS column_name,
            conrel.relname AS ftable,
            conatt.attname AS fcol,
            refn.nspname AS table_schema,
            conn.nspname AS fschema
        FROM pg_constraint c
        JOIN pg_namespace refn ON refn.oid = c.connamespace
        JOIN pg_class refcon ON refcon.oid = c.conrelid
        JOIN pg_namespace conn ON conn.oid = (SELECT relnamespace FROM pg_class WHERE oid = c.confrelid)
        JOIN pg_class conrel ON conrel.oid = c.confrelid
        JOIN unnest(c.conkey) WITH ORDINALITY AS srckeys(attnum, ord) ON TRUE
        JOIN unnest(c.confkey) WITH ORDINALITY AS tgtkeys(attnum, ord) ON srckeys.ord = tgtkeys.ord
        JOIN pg_attribute refatt ON refatt.attrelid = c.conrelid AND refatt.attnum = srckeys.attnum
        JOIN pg_attribute conatt ON conatt.attrelid = c.confrelid AND conatt.attnum = tgtkeys.attnum
        WHERE c.contype = 'f'
    """
    cur = conn.cursor()
    if schema:
        cur.execute(query + " AND refn.nspname = %s ORDER BY 5, 1, srckeys.ord", (schema,))
    else:
        cur.execute(query + " ORDER BY 5, 1, srckeys.ord")
    return cur.fetchall()


def clone_and_strip(src, messy=False):
    """Clone src -> src_bench and drop every FK constraint.

    When messy=True, also rename each child FK column to an opaque name so
    name-based heuristics cannot fire; only data evidence remains. Returns
    (dst, fks, col_renames) where col_renames maps (schema, table, old_col)
    -> new_col for the messy case (empty otherwise)."""
    import psycopg2
    admin = psycopg2.connect(ADMIN)
    admin.autocommit = True
    cur = admin.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (src,))
    if not cur.fetchone():
        admin.close()
        return None
    dst = f"{src}_bench"
    cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (dst,))
    if cur.fetchone():
        cur.execute(f'DROP DATABASE "{dst}"')
    cur.execute(f'CREATE DATABASE "{dst}" TEMPLATE "{src}"')
    admin.close()

    conn = psycopg2.connect(f"postgresql://postgres:12345@localhost:5432/{dst}")
    conn.autocommit = True
    cur = conn.cursor()
    fks = list_fks(conn)

    col_renames = {}
    if messy and fks:
        seen = set()
        idx = 0
        for t, col, ft, fc, sch, fsch in fks:
            key = (sch, t, col)
            if key in seen:
                continue
            seen.add(key)
            idx += 1
            new_col = f"fk_c{idx}"
            # Avoid colliding with an existing column name in the table.
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema=%s AND table_name=%s AND column_name=%s",
                (sch, t, new_col),
            )
            while cur.fetchone():
                idx += 1
                new_col = f"fk_c{idx}"
            try:
                cur.execute(
                    f'ALTER TABLE "{sch}"."{t}" RENAME COLUMN "{col}" TO "{new_col}"'
                )
                col_renames[(sch, t, col)] = new_col
            except Exception:
                pass

    cur.execute(
        """
        SELECT tc.table_schema, tc.table_name, tc.constraint_name
        FROM information_schema.table_constraints tc
        WHERE tc.constraint_type = 'FOREIGN KEY'
        """,
    )
    for sch, t, name in cur.fetchall():
        try:
            cur.execute(f'ALTER TABLE "{sch}"."{t}" DROP CONSTRAINT "{name}"')
        except Exception:
            pass
    conn.close()
    return dst, fks, col_renames


def edge_set(mapping, col_renames=None):
    """Build the (source_table, column, target_table, ref_column) edge set,
    translating messy-renamed columns back to their original names.

    `col_renames` maps (schema, table, old_col) -> new_col; inferred edges carry
    the new (opaque) column names, so we look up the reverse mapping."""
    rev = {}
    for (sch, t, old), new in (col_renames or {}).items():
        rev[(sch, t, new)] = old
        rev[(t, new)] = old  # single-schema inference uses unqualified names
    out = set()
    for r in mapping.get("inferred_relationships", []):
        src = r.get("table")
        col = r.get("column")
        tgt = r.get("references_table")
        ref = r.get("references_column")
        if rev and isinstance(col, str):
            parts = str(src).split(".")
            if len(parts) == 2:
                col = rev.get((parts[0], parts[1], col), col)
            else:
                col = rev.get(("public", parts[0], col), col)
        if rev and isinstance(ref, str):
            parts = str(tgt).split(".")
            if len(parts) == 2:
                ref = rev.get((parts[0], parts[1], ref), ref)
            else:
                ref = rev.get(("public", parts[0], ref), ref)
        out.add((src, str(col), tgt, str(ref)))
    return out


def run(db, col_renames=None):
    cfg = make_cfg(db)
    conn = schema_agent.get_connection(type("C", (), cfg)())
    try:
        mode = schema_agent.detect_schema_mode(conn)
        multi = mode["mode"] == "multi"
        if multi:
            mapping = schema_agent.build_schema_mapping_all(
                conn, db_type="postgresql", llm=None, database_name=db,
            )
        else:
            mapping = schema_agent.build_schema_mapping(
                conn, "public", db_type="postgresql",
            )
    finally:
        conn.close()
    return mapping, multi


def declared_edges(fks, multi):
    """Declared edges as (qualified_source, col, qualified_target, ref_col).

    For single-schema databases the inference emits unqualified table names
    (the same as the declared sets they are compared against), so the schema
    prefix is dropped; multi-schema sets are always schema-qualified."""
    out = set()
    for t, c, ft, fc, sch, fsch in fks:
        if multi:
            out.add((f"{sch}.{t}", str(c), f"{fsch}.{ft}", str(fc)))
        else:
            out.add((t, str(c), ft, str(fc)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbs", nargs="*", default=DEFAULT_DBS)
    ap.add_argument("--messy", action="store_true",
                    help="rename child FK columns to opaque names to force data-only inference")
    args = ap.parse_args()

    rows = []
    totals = {"tp": 0, "fp": 0, "fn": 0}

    for db in args.dbs:
        print("=" * 74)
        print(f"DB: {db} {'[MESSY-NAMING]' if args.messy else ''}", flush=True)
        result = clone_and_strip(db, messy=args.messy)
        if result is None:
            print("  skipped (not found)")
            continue
        dst, declared, col_renames = result
        t0 = time.time()
        try:
            stripped_map, multi = run(dst, col_renames)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue
        elapsed = time.time() - t0
        declared_set = declared_edges(declared, multi)

        inferred = edge_set(stripped_map, col_renames)
        correct = declared_set & inferred
        wrong = inferred - declared_set
        missed = declared_set - inferred
        tp = len(correct)
        fp = len(wrong)
        fn = len(missed)

        precision = tp / len(inferred) if inferred else 0.0
        recall = tp / len(declared_set) if declared_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        fpr = fp / len(inferred) if inferred else 0.0
        fnr = fn / len(declared_set) if declared_set else 0.0

        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

        print(f"  declared={len(declared_set)} inferred={len(inferred)} ({elapsed:.1f}s)")
        print(f"  Precision={precision:.1%} Recall={recall:.1%} F1={f1:.3f} FPR={fpr:.1%} FNR={fnr:.1%}")
        if correct:
            print("  CORRECT:")
            for e in sorted(correct, key=str):
                print(f"    {'.'.join([e[0], e[1]])} -> {'.'.join([e[2], e[3]])}")
        if wrong:
            print("  FALSE POSITIVES:")
            for e in sorted(wrong, key=str)[:20]:
                print(f"    {'.'.join([e[0], e[1]])} -> {'.'.join([e[2], e[3]])}")
            if len(wrong) > 20:
                print(f"    ... and {len(wrong) - 20} more")
        if missed:
            print("  FALSE NEGATIVES (missed):")
            for e in sorted(missed, key=str)[:20]:
                print(f"    {'.'.join([e[0], e[1]])} -> {'.'.join([e[2], e[3]])}")
            if len(missed) > 20:
                print(f"    ... and {len(missed) - 20} more")
        rows.append({
            "db": db, "declared": len(declared_set), "inferred": len(inferred),
            "correct": tp, "false_positives": fp, "missed": fn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "fpr": round(fpr, 4), "fnr": round(fnr, 4),
            "elapsed_s": round(elapsed, 1),
        })

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    print("\n" + "=" * 74)
    print(f"OVERALL (across {len(rows)} DBs): Precision={p:.1%} Recall={r:.1%} F1={f1:.3f} (TP={tp} FP={fp} FN={fn})")
    out = Path(__file__).resolve().parent / "benchmark_results.json"
    out.write_text(json.dumps({"dbs": rows, "overall": {
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
    }}, indent=2), encoding="utf-8")
    print(f"Saved {out.name}")


if __name__ == "__main__":
    main()