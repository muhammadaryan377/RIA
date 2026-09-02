from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pandas.errors import EmptyDataError, ParserError


class CsvValidationError(ValueError):
    """Raised when a CSV file cannot be safely interpreted as a table."""


@dataclass(frozen=True)
class CsvDialectInfo:
    delimiter: str
    quotechar: str
    doublequote: bool
    escapechar: str | None
    skipinitialspace: bool


class CsvSchemaExtractor:
    """
    Deterministic CSV ingestion + schema profiling for ARIA.

    Responsibilities:
    - validate a CSV file before ingestion
    - detect encoding and CSV dialect
    - preserve raw values as strings during ingestion
    - profile row/column structure and data quality
    - conservatively infer logical column types
    - expose candidate keys without pretending they are declared primary keys
    - create a stable file fingerprint for change detection

    This class intentionally does NOT use an LLM for structural CSV parsing.
    An LLM can be added later for semantic enrichment of ambiguous column names.
    """

    DEFAULT_MISSING_TOKENS = frozenset(
        {"", "null", "none", "nan", "missing"}
    )
    CANDIDATE_DELIMITERS = ",;\t|"

    def __init__(
        self,
        file_path: str | Path,
        *,
        sample_values: int = 5,
        max_file_size_mb: int | None = 100,
        missing_tokens: Iterable[str] | None = None,
    ) -> None:
        self.path = Path(file_path)
        self.sample_values = max(1, int(sample_values))
        self.max_file_size_mb = max_file_size_mb
        self.missing_tokens = {
            token.strip().lower()
            for token in (missing_tokens or self.DEFAULT_MISSING_TOKENS)
        }

        self._encoding: str | None = None
        self._dialect: CsvDialectInfo | None = None
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> dict[str, Any]:
        """Parse, validate and return a JSON-serialisable schema profile."""
        self._validate_file()
        self._encoding = self._detect_encoding()
        self._dialect = self._detect_dialect(self._encoding)
        self._validate_header(self._encoding, self._dialect)
        self._df = self._read_dataframe(self._encoding, self._dialect)

        table_name = self._safe_table_name(self.path.stem)
        row_count = len(self._df)

        columns = [
            self._profile_column(self._df[column_name], column_name, row_count)
            for column_name in self._df.columns
        ]

        duplicate_rows = int(self._df.duplicated().sum()) if row_count else 0
        duplicate_pct = round((duplicate_rows / row_count) * 100, 2) if row_count else 0.0

        warnings: list[str] = []
        if row_count == 0:
            warnings.append("CSV contains headers but no data rows.")
        if duplicate_rows:
            warnings.append(f"{duplicate_rows} duplicate row(s) detected.")
        for column in columns:
            name = column["name"]
            if column["null_percentage"] > 50:
                warnings.append(f"Column '{name}' contains more than 50% missing values.")
            if column["inferred_type"] == "UNKNOWN":
                warnings.append(f"Column '{name}' has no usable values.")
            if column["distinct_count"] == 1 and column["non_null_count"] > 1:
                warnings.append(f"Column '{name}' is constant.")
            if column["whitespace_value_count"]:
                warnings.append(
                    f"Column '{name}' has {column['whitespace_value_count']} value(s) "
                    "with surrounding whitespace."
                )

        return {
            "source_type": "csv",
            "source_name": self.path.name,
            "table_name": table_name,
            "file": {
                "path": str(self.path),
                "size_bytes": self.path.stat().st_size,
                "sha256": self._sha256(),
                "modified_at_utc": datetime.fromtimestamp(
                    self.path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
                "encoding": self._encoding,
                "delimiter": self._dialect.delimiter,
                "quotechar": self._dialect.quotechar,
            },
            "row_count": row_count,
            "column_count": len(self._df.columns),
            "columns": columns,
            "quality": {
                "duplicate_rows": duplicate_rows,
                "duplicate_row_percentage": duplicate_pct,
                "warnings": warnings,
            },
        }

    def load_dataframe(self) -> pd.DataFrame:
        """
        Return a defensive copy of the parsed DataFrame.

        The file is parsed if extract() has not been called yet. Values are kept
        as strings so identifiers such as 00123 are not silently converted to 123.
        """
        if self._df is None:
            self.extract()
        assert self._df is not None
        return self._df.copy()

    def save_schema_json(self, output_path: str | Path) -> Path:
        """Write the extracted profile to JSON and return the output path."""
        payload = self.extract()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output

    # ------------------------------------------------------------------
    # File and dialect validation
    # ------------------------------------------------------------------

    def _validate_file(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.path}")
        if not self.path.is_file():
            raise CsvValidationError(f"Expected a file, got: {self.path}")
        if self.path.suffix.lower() != ".csv":
            raise CsvValidationError("Only .csv files are accepted by CsvSchemaExtractor.")

        size = self.path.stat().st_size
        if size == 0:
            raise CsvValidationError("CSV file is empty.")

        if self.max_file_size_mb is not None:
            max_bytes = self.max_file_size_mb * 1024 * 1024
            if size > max_bytes:
                raise CsvValidationError(
                    f"CSV is {size / (1024 * 1024):.1f} MB; "
                    f"configured limit is {self.max_file_size_mb} MB. "
                    "Use chunked ingestion for larger files."
                )

        # CSV is text. A NUL byte is a strong signal that the upload is binary
        # or otherwise not a normal text CSV.
        with self.path.open("rb") as fh:
            if b"\x00" in fh.read(8192):
                # UTF-16 legitimately contains NUL bytes, so defer if BOM exists.
                with self.path.open("rb") as prefix_fh:
                    raw_prefix = prefix_fh.read(4)
                utf16_boms = (b"\xff\xfe", b"\xfe\xff")
                if not raw_prefix.startswith(utf16_boms):
                    raise CsvValidationError("File appears to be binary, not a normal CSV text file.")

    def _detect_encoding(self) -> str:
        # Read only the sample needed for detection. read_bytes() would load the
        # complete file before slicing it, which is wasteful for large uploads.
        with self.path.open("rb") as fh:
            raw = fh.read(65536)

        if raw.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            return "utf-16"

        for encoding in ("utf-8", "cp1252"):
            try:
                raw.decode(encoding)
                return encoding
            except UnicodeDecodeError:
                continue

        # latin-1 maps every byte. It is a safe final decoding fallback, not a
        # claim that the source was semantically authored in Latin-1.
        return "latin-1"

    def _sample_text(self, encoding: str, char_limit: int = 65536) -> str:
        with self.path.open("r", encoding=encoding, errors="strict", newline="") as fh:
            return fh.read(char_limit)

    def _detect_dialect(self, encoding: str) -> CsvDialectInfo:
        sample = self._sample_text(encoding)
        if not sample.strip():
            raise CsvValidationError("CSV contains no readable text.")

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=self.CANDIDATE_DELIMITERS)
            delimiter = dialect.delimiter
            quotechar = dialect.quotechar or '"'
            doublequote = bool(dialect.doublequote)
            escapechar = dialect.escapechar
            skipinitialspace = bool(dialect.skipinitialspace)
        except csv.Error:
            # A valid one-column CSV has no delimiter to sniff.
            delimiter = ","
            quotechar = '"'
            doublequote = True
            escapechar = None
            skipinitialspace = False

        return CsvDialectInfo(
            delimiter=delimiter,
            quotechar=quotechar,
            doublequote=doublequote,
            escapechar=escapechar,
            skipinitialspace=skipinitialspace,
        )

    def _validate_header(self, encoding: str, dialect: CsvDialectInfo) -> None:
        try:
            with self.path.open("r", encoding=encoding, errors="strict", newline="") as fh:
                reader = csv.reader(
                    fh,
                    delimiter=dialect.delimiter,
                    quotechar=dialect.quotechar,
                    doublequote=dialect.doublequote,
                    escapechar=dialect.escapechar,
                    skipinitialspace=dialect.skipinitialspace,
                )
                header = next(reader)
        except (StopIteration, csv.Error) as exc:
            raise CsvValidationError("CSV header could not be read.") from exc

        if not header:
            raise CsvValidationError("CSV header is empty.")

        cleaned = [name.strip() for name in header]
        if any(not name for name in cleaned):
            raise CsvValidationError("CSV contains one or more blank column names.")

        lowered = [name.casefold() for name in cleaned]
        duplicates = sorted({name for name in lowered if lowered.count(name) > 1})
        if duplicates:
            raise CsvValidationError(
                "Duplicate CSV column names are not allowed: " + ", ".join(duplicates)
            )

    def _read_dataframe(self, encoding: str, dialect: CsvDialectInfo) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                self.path,
                sep=dialect.delimiter,
                quotechar=dialect.quotechar,
                doublequote=dialect.doublequote,
                escapechar=dialect.escapechar,
                skipinitialspace=dialect.skipinitialspace,
                encoding=encoding,
                encoding_errors="strict",
                dtype="string",
                keep_default_na=False,
                on_bad_lines="error",
                low_memory=False,
            )
        except EmptyDataError as exc:
            raise CsvValidationError("CSV contains no tabular data.") from exc
        except (ParserError, UnicodeDecodeError) as exc:
            raise CsvValidationError(f"CSV parsing failed: {exc}") from exc

        # Header whitespace is formatting noise; data values are left untouched.
        df.columns = [str(column).strip() for column in df.columns]

        # Re-check because pandas can alter duplicate names during parsing.
        lowered = [column.casefold() for column in df.columns]
        if len(lowered) != len(set(lowered)):
            raise CsvValidationError("CSV column names are not unique after parsing.")

        return df

    # ------------------------------------------------------------------
    # Profiling and type inference
    # ------------------------------------------------------------------

    def _profile_column(
        self,
        series: pd.Series,
        column_name: str,
        row_count: int,
    ) -> dict[str, Any]:
        string_series = series.astype("string")
        trimmed = string_series.str.strip()
        missing_mask = trimmed.str.lower().isin(self.missing_tokens) | trimmed.isna()
        non_missing = trimmed[~missing_mask]
        original_non_missing = string_series[~missing_mask]

        null_count = int(missing_mask.sum())
        distinct_count = int(non_missing.nunique(dropna=True))
        non_null_count = int(len(non_missing))
        whitespace_value_count = int((original_non_missing != non_missing).sum())

        null_percentage = round((null_count / row_count) * 100, 2) if row_count else 0.0
        distinct_percentage = (
            round((distinct_count / row_count) * 100, 2) if row_count else 0.0
        )
        is_unique = non_null_count > 0 and distinct_count == non_null_count
        # Uniqueness alone is not enough: prices and names may be accidentally
        # unique in a small sample. Only identifier-like columns are suggested.
        identifier_name = self._is_identifier_like(column_name)
        candidate_key = is_unique and null_count == 0 and identifier_name

        sample_values = [
            self._json_scalar(value)
            for value in non_missing.drop_duplicates().head(self.sample_values).tolist()
        ]

        inferred_type, inference_note = self._infer_logical_type(non_missing.tolist())
        statistics = self._column_statistics(non_missing, inferred_type)

        profile = {
            "name": column_name,
            "inferred_type": inferred_type,
            "inference_note": inference_note,
            "nullable": null_count > 0,
            "non_null_count": non_null_count,
            "null_count": null_count,
            "null_percentage": null_percentage,
            "distinct_count": distinct_count,
            "distinct_percentage": distinct_percentage,
            "is_unique": is_unique,
            "candidate_key": candidate_key,
            "whitespace_value_count": whitespace_value_count,
            "sample_values": sample_values,
        }
        if statistics is not None:
            profile["statistics"] = statistics
        return profile

    def _column_statistics(
        self,
        values: pd.Series,
        inferred_type: str,
    ) -> dict[str, Any] | None:
        """Return a small, useful summary appropriate for the logical type."""
        if values.empty:
            return None

        if inferred_type in {"INTEGER", "DECIMAL"}:
            numeric = pd.to_numeric(values, errors="coerce")
            return {
                "min": self._json_number(numeric.min()),
                "max": self._json_number(numeric.max()),
                "mean": self._json_number(numeric.mean(), rounded=True),
                "median": self._json_number(numeric.median(), rounded=True),
            }

        if inferred_type in {"DATE", "DATETIME"}:
            parsed = pd.to_datetime(values, errors="coerce", utc=inferred_type == "DATETIME")
            return {
                "earliest": parsed.min().isoformat(),
                "latest": parsed.max().isoformat(),
            }

        # sample_values already explains string and boolean columns. Repeating
        # those values with counts makes the schema noisy and redundant.
        return None

    @staticmethod
    def _json_number(value: Any, *, rounded: bool = False) -> int | float:
        number = float(value)
        if number.is_integer():
            return int(number)
        return round(number, 4) if rounded else number

    @staticmethod
    def _json_scalar(value: Any) -> Any:
        if value is pd.NA:
            return None
        # Values are strings in this extractor; str() keeps this robust if the
        # DataFrame backend changes later.
        return str(value)

    @staticmethod
    def _is_identifier_like(column_name: str) -> bool:
        """Recognise common identifier suffixes across typical naming styles."""
        separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", column_name.strip())
        normalised = re.sub(r"[^0-9a-zA-Z]+", "_", separated).strip("_").lower()
        return bool(re.search(r"(?:^|_)(?:id|key|code)$", normalised))

    def _infer_logical_type(self, values: list[Any]) -> tuple[str, str | None]:
        cleaned = [str(value).strip() for value in values if str(value).strip()]
        if not cleaned:
            return "UNKNOWN", "No non-missing values are available for inference."

        lowered = {value.casefold() for value in cleaned}
        bool_tokens = {"true", "false", "yes", "no", "y", "n"}
        if lowered.issubset(bool_tokens) and len(lowered) >= 1:
            return "BOOLEAN", None

        integer_re = re.compile(r"^[+-]?\d+$")
        if all(integer_re.fullmatch(value) for value in cleaned):
            unsigned = [value.lstrip("+-") for value in cleaned]
            if any(len(value) > 1 and value.startswith("0") for value in unsigned):
                return (
                    "STRING",
                    "Numeric-looking values contain leading zeros; kept as STRING to preserve identifiers.",
                )
            return "INTEGER", None

        decimal_re = re.compile(r"^[+-]?(?:\d+\.\d+|\d+\.|\.\d+|\d+[eE][+-]?\d+|\d+\.\d+[eE][+-]?\d+)$")
        numeric_re = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
        if all(numeric_re.fullmatch(value) for value in cleaned) and any(
            decimal_re.fullmatch(value) for value in cleaned
        ):
            return "DECIMAL", None

        # Date inference is intentionally conservative. Only ISO-like values
        # are auto-classified so 03/04/2026 is not guessed as DD/MM vs MM/DD.
        if all(self._is_iso_datetime(value) for value in cleaned):
            if any("T" in value or " " in value for value in cleaned):
                return "DATETIME", None
            return "DATE", None

        return "STRING", None

    @staticmethod
    def _is_iso_datetime(value: str) -> bool:
        candidate = value.strip()
        if not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?",
            candidate,
        ):
            return False

        try:
            datetime.fromisoformat(candidate.replace("Z", "+00:00"))
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_table_name(name: str) -> str:
        normalised = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_").lower()
        if not normalised:
            return "csv_table"
        if normalised[0].isdigit():
            normalised = f"t_{normalised}"
        return normalised

    def _sha256(self) -> str:
        digest = hashlib.sha256()
        with self.path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


class CsvFolderSchemaExtractor:
    """Extract all CSV files in one folder as independent tabular sources."""

    def __init__(self, folder_path: str | Path, **csv_options: Any) -> None:
        self.folder = Path(folder_path)
        self.csv_options = csv_options

    def extract(self) -> dict[str, Any]:
        if not self.folder.exists() or not self.folder.is_dir():
            raise CsvValidationError(f"CSV folder not found: {self.folder}")

        files = sorted(
            path for path in self.folder.iterdir()
            if path.is_file() and path.suffix.lower() == ".csv"
        )
        if not files:
            raise CsvValidationError(f"No .csv files found in: {self.folder}")

        tables = [CsvSchemaExtractor(path, **self.csv_options).extract() for path in files]
        return {
            "source_type": "csv_collection",
            "folder": str(self.folder),
            "table_count": len(tables),
            "tables": tables,
        }
