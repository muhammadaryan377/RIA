"""Data source connection routes: relational databases (PostgreSQL / MySQL) + CSV files."""

import json
import re
import types
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

import csv_orchestrator
import schema_agent
from core.config import DB_TYPES, PROVIDER_NAMES, SOURCE_TYPES, SCHEMA_DIR, PROCESSED_DIR, INSIGHTS_DIR, DATA_DIR, get_session
from core.db import build_db_uri
from core.deps import require_writable
from csv_handler import CsvValidationError
from llm_provider import create_provider

router = APIRouter()


class ProviderSwitchRequest(BaseModel):
    provider: str = "local"
    api_key: str | None = None      # optional key override for hosted providers
    base_url: str | None = None     # optional endpoint override for hosted providers


@router.post("/api/provider/switch")
def switch_provider(request: ProviderSwitchRequest, user: dict = Depends(require_writable)):
    """Explicitly switch the LLM backend for this session."""
    session = get_session(user["user_id"])
    if request.provider not in PROVIDER_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{request.provider}'. Choose from {sorted(PROVIDER_NAMES)}.")
    current = session.get("provider_name")
    if current == request.provider:
        return {"ok": True, "provider": current, "changed": False}

    try:
        llm = create_provider(
            provider=request.provider,
            api_key=request.api_key,
            base_url=request.base_url,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not initialise '{request.provider}' LLM provider: {exc}")

    session["provider"] = llm
    session["provider_name"] = request.provider
    return {"ok": True, "provider": request.provider, "changed": True}


class ConnectRequest(BaseModel):
    provider: str = "local"
    source_type: str = "relational"
    db_type: str = "postgresql"   # 'postgresql' | 'mysql'
    host: str = "localhost"
    port: int | str = 5432
    db: str | None = None
    user: str | None = None
    password: str | None = None
    db_schema: str = "public"


def _do_connect(request: ConnectRequest, user: dict, db_type: str) -> dict:
    """Shared connect logic. `db_type` is authoritative (set by the calling endpoint)."""
    session = get_session(user["user_id"])
    if request.provider not in PROVIDER_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{request.provider}'. Choose from {sorted(PROVIDER_NAMES)}.")
    if request.source_type not in SOURCE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown source type '{request.source_type}'. Choose from {sorted(SOURCE_TYPES)}.")
    if request.source_type == "relational" and db_type not in DB_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown database type '{db_type}'. Choose from {sorted(DB_TYPES)}.")

    if request.source_type == "relational":
        if not (request.db and request.user and request.password):
            raise HTTPException(status_code=400, detail="Relational source requires db, user, and password.")

    try:
        llm = create_provider(provider=request.provider)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not initialise '{request.provider}' LLM provider: {exc}")

    session["provider"] = llm
    session["provider_name"] = request.provider
    session["source_type"] = request.source_type
    session["db"] = None
    session["db_type"] = db_type
    session["db_uri"] = None
    session["db_name"] = None
    uid_dir = f"user_{user['user_id']}"
    session["schema_path"] = SCHEMA_DIR / uid_dir / "schema_mapping_latest.json"
    session["processed_path"] = PROCESSED_DIR / uid_dir / "processed_data.json"
    session["insights_path"] = INSIGHTS_DIR / uid_dir / "insights.json"
    session["processed_path"].parent.mkdir(parents=True, exist_ok=True)
    session["insights_path"].parent.mkdir(parents=True, exist_ok=True)

    config = request.model_dump()
    config["db_type"] = db_type
    conn = None
    try:
        config["port"] = int(config["port"])
        conn = schema_agent.get_connection(types.SimpleNamespace(**config))
        with conn.cursor() as cur:
            if db_type == "mysql":
                cur.execute("SELECT DATABASE();")
                db_name = cur.fetchone()[0]
                current_db_schema = db_name
            else:
                cur.execute("SELECT current_database(), current_schema();")
                db_name, current_db_schema = cur.fetchone()
    except SystemExit as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid port number.")
    except Exception as exc:
        detail = f"Connection failed: {exc}"
        if db_type == "mysql":
            detail += (" Hint: is the MySQL server running, and is pymysql installed "
                       "(pip install pymysql)? Verify host/port, database, user, password.")
        else:
            detail += (" Hint: is the PostgreSQL service running on that host/port? "
                       "Verify the database name, user, and password are correct.")
        raise HTTPException(status_code=400, detail=detail)
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    from core.secret_box import encrypt as _encrypt_secret
    safe_config = dict(config)
    if "password" in safe_config and safe_config["password"] is not None:
        safe_config["password"] = _encrypt_secret(str(safe_config["password"]))
    session["db"] = safe_config
    session["db_type"] = db_type
    session["db_uri"] = build_db_uri(config, db_type)
    session["db_name"] = db_name
    session["dialect"] = db_type

    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", db_name)
    uid_dir = f"user_{user['user_id']}"
    session["schema_path"] = SCHEMA_DIR / uid_dir / "schema_mapping_latest.json"
    session["processed_path"] = PROCESSED_DIR / uid_dir / f"processed_data_{safe}.json"
    session["insights_path"] = INSIGHTS_DIR / uid_dir / f"insights_{safe}.json"
    session["processed_path"].parent.mkdir(parents=True, exist_ok=True)
    session["insights_path"].parent.mkdir(parents=True, exist_ok=True)

    return {
        "ok": True,
        "provider": request.provider,
        "source_type": "relational",
        "database": db_name,
        "schema": current_db_schema,
        "db_type": db_type,
        "schema_file": session["schema_path"].name,
    }


@router.post("/api/connect")
def connect(request: ConnectRequest, user: dict = Depends(require_writable)):
    """Connect to a source; database type taken from the request body."""
    return _do_connect(request, user, request.db_type)


@router.post("/api/connect/postgresql")
def connect_postgresql(request: ConnectRequest, user: dict = Depends(require_writable)):
    """Connect to a PostgreSQL source (explicit route)."""
    return _do_connect(request, user, "postgresql")


@router.post("/api/connect/mysql")
def connect_mysql(request: ConnectRequest, user: dict = Depends(require_writable)):
    """Connect to a MySQL source (explicit route)."""
    return _do_connect(request, user, "mysql")


@router.post("/api/connect_file")
def connect_file(provider: str = Form("local"), file: UploadFile = File(...),
                 llm_api_key: str | None = Form(None),
                 llm_base_url: str | None = Form(None),
                 user: dict = Depends(require_writable)):
    """Connect a semi-structured source: upload a CSV file.

    The file is validated and profiled deterministically with the CSV handler
    (encoding/dialect detection, header checks, type inference, data-quality
    warnings), its schema mapping is written in the Goal Agent's ``tables``
    format, the rows are loaded into a file-based SQLite database, and the
    session is pointed at it so the Goal Agent and Insight Agent consume it
    exactly like a relational source (dialect=sqlite).
    """
    if provider not in PROVIDER_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{provider}'. Choose from {sorted(PROVIDER_NAMES)}.")
    filename = (file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="No file received.")
    ext = Path(filename).suffix.lower()
    if ext != ".csv":
        raise HTTPException(
            status_code=400,
            detail="Only .csv files are accepted for file sources right now. "
                   "Tabular PDF extraction is not available in this build.",
        )
    try:
        llm = create_provider(provider=provider, api_key=llm_api_key, base_url=llm_base_url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not initialise '{provider}' LLM provider: {exc}")

    upload_dir = DATA_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(filename).stem)[:64] or "upload"
    file_path = upload_dir / f"{safe_name}_{stamp}{ext}"
    file_path.write_bytes(file.file.read())

    try:
        table_name = csv_orchestrator.safe_table_name(safe_name)
        mapping = csv_orchestrator.build_csv_schema_mapping(file_path, table_name=table_name)
        sqlite_path = upload_dir / f"{safe_name}_{stamp}.db"
        db_uri, table_name, row_count = csv_orchestrator.load_csv_to_sqlite(
            file_path, sqlite_path, table_name=table_name
        )
    except CsvValidationError as exc:
        # Deterministic validation failed; the upload is rejected as-is.
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}")
    except HTTPException:
        raise
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"CSV processing failed: {exc}")

    uid_dir = f"user_{user['user_id']}"
    safe_db_name = mapping["database"]
    schema_path = SCHEMA_DIR / uid_dir / f"schema_mapping_{safe_db_name}_latest.json"
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    session = get_session(user["user_id"])
    session["provider"] = llm
    session["provider_name"] = provider
    session["source_type"] = "semi-structured"
    session["db"] = {".csv": str(file_path)}
    session["db_type"] = "sqlite"
    session["db_uri"] = db_uri
    session["db_name"] = table_name
    session["schema_path"] = schema_path
    session["processed_path"] = PROCESSED_DIR / uid_dir / f"processed_data_{safe_db_name}.json"
    session["insights_path"] = INSIGHTS_DIR / uid_dir / f"insights_{safe_db_name}.json"
    session["processed_path"].parent.mkdir(parents=True, exist_ok=True)
    session["insights_path"].parent.mkdir(parents=True, exist_ok=True)
    session["dialect"] = "sqlite"
    session["file_type"] = "csv"

    columns = [c["column"] for c in mapping["tables"][table_name]["columns"]]
    quality = mapping["tables"][table_name]["csv_profile"]["quality"]
    return {
        "ok": True,
        "provider": provider,
        "source_type": "semi-structured",
        "file": filename,
        "type": "csv",
        "tables": {table_name: row_count},
        "table": table_name,
        "rows": row_count,
        "columns": columns,
        "schema_file": schema_path.name,
        "quality": quality,
        "db_file": sqlite_path.name,
    }



