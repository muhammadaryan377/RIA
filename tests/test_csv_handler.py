from pathlib import Path

import pytest

from csv_handler import CsvFolderSchemaExtractor, CsvSchemaExtractor, CsvValidationError


def write(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.write_text(text, encoding=encoding, newline="")
    return path


def test_basic_profile_and_types(tmp_path: Path):
    path = write(
        tmp_path / "sales.csv",
        "order_id,sku,amount,active,order_date\n"
        "1,00123,10.50,true,2026-08-01\n"
        "2,00456,20.00,false,2026-08-02\n"
        "3,00789,,true,2026-08-03\n",
    )

    schema = CsvSchemaExtractor(path).extract()
    cols = {c["name"]: c for c in schema["columns"]}

    assert schema["table_name"] == "sales"
    assert schema["row_count"] == 3
    assert cols["order_id"]["inferred_type"] == "INTEGER"
    assert cols["order_id"]["candidate_key"] is True
    assert cols["sku"]["inferred_type"] == "STRING"
    assert "leading zeros" in cols["sku"]["inference_note"]
    assert cols["amount"]["inferred_type"] == "DECIMAL"
    assert cols["amount"]["null_count"] == 1
    assert cols["active"]["inferred_type"] == "BOOLEAN"
    assert cols["order_date"]["inferred_type"] == "DATE"


def test_semicolon_and_quoted_delimiter(tmp_path: Path):
    path = write(
        tmp_path / "customers.csv",
        'id;name;note\n1;Ali;"likes tea; coffee"\n2;Sara;ok\n',
    )

    schema = CsvSchemaExtractor(path).extract()
    assert schema["file"]["delimiter"] == ";"
    assert schema["column_count"] == 3
    assert schema["row_count"] == 2


def test_cp1252_fallback(tmp_path: Path):
    path = tmp_path / "names.csv"
    path.write_bytes("id,name\n1,André\n".encode("cp1252"))

    schema = CsvSchemaExtractor(path).extract()
    assert schema["file"]["encoding"] in {"cp1252", "latin-1"}
    assert schema["row_count"] == 1


def test_duplicate_header_rejected(tmp_path: Path):
    path = write(tmp_path / "bad.csv", "id,ID\n1,2\n")
    with pytest.raises(CsvValidationError, match="Duplicate CSV column names"):
        CsvSchemaExtractor(path).extract()


def test_blank_header_rejected(tmp_path: Path):
    path = write(tmp_path / "bad.csv", "id,,amount\n1,x,10\n")
    with pytest.raises(CsvValidationError, match="blank column names"):
        CsvSchemaExtractor(path).extract()


def test_malformed_row_rejected(tmp_path: Path):
    path = write(tmp_path / "bad.csv", "id,name\n1,A\n2,B,extra\n")
    with pytest.raises(CsvValidationError, match="CSV parsing failed"):
        CsvSchemaExtractor(path).extract()


def test_duplicate_rows_quality_signal(tmp_path: Path):
    path = write(tmp_path / "dup.csv", "id,name\n1,A\n1,A\n2,B\n")
    schema = CsvSchemaExtractor(path).extract()
    assert schema["quality"]["duplicate_rows"] == 1
    assert schema["quality"]["duplicate_row_percentage"] == 33.33


def test_ambiguous_date_is_not_guessed(tmp_path: Path):
    path = write(tmp_path / "dates.csv", "id,date\n1,03/04/2026\n2,04/05/2026\n")
    schema = CsvSchemaExtractor(path).extract()
    cols = {c["name"]: c for c in schema["columns"]}
    assert cols["date"]["inferred_type"] == "STRING"


def test_folder_extractor(tmp_path: Path):
    write(tmp_path / "a.csv", "id\n1\n")
    write(tmp_path / "b.csv", "id\n2\n")

    schema = CsvFolderSchemaExtractor(tmp_path).extract()
    assert schema["table_count"] == 2


def test_sha256_is_stable(tmp_path: Path):
    path = write(tmp_path / "x.csv", "id\n1\n")
    first = CsvSchemaExtractor(path).extract()["file"]["sha256"]
    second = CsvSchemaExtractor(path).extract()["file"]["sha256"]
    assert first == second


def test_useful_statistics_and_conservative_keys(tmp_path: Path):
    path = write(
        tmp_path / "sales.csv",
        "order_id,product_name,amount,ordered_on,category\n"
        "1,Alpha,10.00,2026-08-01,A\n"
        "2,Beta,20.00,2026-08-03,A\n"
        "3,Gamma,30.00,2026-08-02,B\n",
    )

    schema = CsvSchemaExtractor(path).extract()
    cols = {column["name"]: column for column in schema["columns"]}

    assert cols["order_id"]["candidate_key"] is True
    assert cols["product_name"]["candidate_key"] is False
    assert cols["amount"]["statistics"] == {
        "min": 10,
        "max": 30,
        "mean": 20,
        "median": 20,
    }
    assert cols["ordered_on"]["statistics"]["earliest"].startswith("2026-08-01")
    assert cols["ordered_on"]["statistics"]["latest"].startswith("2026-08-03")
    assert "statistics" not in cols["category"]
    assert cols["category"]["sample_values"] == ["A", "B"]


def test_missing_tokens_and_column_warnings(tmp_path: Path):
    path = write(
        tmp_path / "quality.csv",
        "id,status,note\n1, active ,missing\n2,active,null\n3,active,value\n",
    )

    schema = CsvSchemaExtractor(path).extract()
    cols = {column["name"]: column for column in schema["columns"]}

    assert cols["note"]["null_count"] == 2
    assert cols["status"]["whitespace_value_count"] == 1
    assert "Column 'status' is constant." in schema["quality"]["warnings"]
    assert any("surrounding whitespace" in warning for warning in schema["quality"]["warnings"])


def test_folder_accepts_uppercase_csv_extension(tmp_path: Path):
    write(tmp_path / "UPPER.CSV", "id\n1\n")
    assert CsvFolderSchemaExtractor(tmp_path).extract()["table_count"] == 1


def test_na_is_preserved_by_default(tmp_path: Path):
    path = write(tmp_path / "regions.csv", "customer_id,region\nC001,NA\nC002,EU\n")
    columns = {c["name"]: c for c in CsvSchemaExtractor(path).extract()["columns"]}

    assert columns["region"]["null_count"] == 0
    assert "NA" in columns["region"]["sample_values"]


def test_custom_missing_token_still_works(tmp_path: Path):
    path = write(tmp_path / "regions.csv", "customer_id,region\nC001,NA\nC002,EU\n")
    columns = {
        c["name"]: c
        for c in CsvSchemaExtractor(path, missing_tokens={"", "na"}).extract()["columns"]
    }

    assert columns["region"]["null_count"] == 1


@pytest.mark.parametrize(
    "identifier_name",
    ["customerId", "CustomerID", "ProductCode", "orderKey", "customer-id", "customer id"],
)
def test_identifier_name_variants_are_candidate_keys(
    tmp_path: Path, identifier_name: str
):
    path = write(tmp_path / "identifiers.csv", f"{identifier_name},name\nX001,A\nX002,B\n")
    columns = {c["name"]: c for c in CsvSchemaExtractor(path).extract()["columns"]}

    assert columns[identifier_name]["candidate_key"] is True
    assert columns["name"]["candidate_key"] is False


def test_unique_metric_is_not_a_candidate_key(tmp_path: Path):
    path = write(tmp_path / "metrics.csv", "unit_price\n10.25\n11.50\n12.75\n")
    column = CsvSchemaExtractor(path).extract()["columns"][0]

    assert column["is_unique"] is True
    assert column["candidate_key"] is False


def test_output_path_is_not_forced_to_absolute(tmp_path: Path, monkeypatch):
    path = write(tmp_path / "portable.csv", "id\n1\n")
    monkeypatch.chdir(tmp_path)

    schema = CsvSchemaExtractor("portable.csv").extract()

    assert schema["file"]["path"] == "portable.csv"
    assert not Path(schema["file"]["path"]).is_absolute()


def test_header_only_csv_is_valid_with_warning(tmp_path: Path):
    path = write(tmp_path / "headers.csv", "order_id,customer_id,amount\n")

    schema = CsvSchemaExtractor(path).extract()

    assert schema["row_count"] == 0
    assert schema["column_count"] == 3
    assert "CSV contains headers but no data rows." in schema["quality"]["warnings"]
