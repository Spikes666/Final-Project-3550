import pandas as pd
import pytest

from datalens.loading import load_dataframe


def test_load_csv(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(path, index=False)

    df = load_dataframe(str(path))

    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_load_json(tmp_path):
    path = tmp_path / "data.json"
    pd.DataFrame({"a": [1, 2]}).to_json(path, orient="records")

    df = load_dataframe(str(path))

    assert list(df["a"]) == [1, 2]


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "data.exe"
    path.write_text("not a dataset")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_dataframe(str(path))
