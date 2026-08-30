"""Load tabular datasets of various file types into pandas DataFrames."""

from __future__ import annotations

import io

import pandas as pd

SUPPORTED_EXTENSIONS = ("csv", "tsv", "xlsx", "xls", "json", "parquet")


def load_dataframe(file) -> pd.DataFrame:
    """Load a DataFrame from an uploaded file-like object or path.

    Dispatches on file extension. `file` may be a path string or an
    object with `.name` and file-like read support (e.g. Streamlit's
    UploadedFile).
    """
    name = getattr(file, "name", str(file))
    ext = name.rsplit(".", 1)[-1].lower()

    if ext == "csv":
        return pd.read_csv(file)
    if ext == "tsv":
        return pd.read_csv(file, sep="\t")
    if ext in ("xlsx", "xls"):
        return pd.read_excel(file)
    if ext == "json":
        return pd.read_json(file)
    if ext == "parquet":
        return pd.read_parquet(file)

    raise ValueError(
        f"Unsupported file type '.{ext}'. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
    )


def dataframe_to_download_bytes(df: pd.DataFrame, ext: str = "csv") -> bytes:
    """Serialize a DataFrame back to bytes for a Streamlit download button."""
    buffer = io.BytesIO()
    if ext == "csv":
        buffer.write(df.to_csv(index=False).encode("utf-8"))
    elif ext == "json":
        buffer.write(df.to_json(orient="records").encode("utf-8"))
    else:
        raise ValueError(f"Unsupported export type '.{ext}'")
    return buffer.getvalue()
