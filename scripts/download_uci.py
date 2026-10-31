"""Download the official UCI Diabetes 130-US Hospitals files into data/raw/."""

from __future__ import annotations

import io
import ssl
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

URLS = [
    "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip",
]
HEADERS = {"User-Agent": "HospitalOperationsProject/1.0 (educational portfolio)"}


def download() -> Path:
    last_error = None
    context = ssl.create_default_context()
    for url in URLS:
        print("Trying", url)
        try:
            req = Request(url, headers=HEADERS)
            with urlopen(req, timeout=120, context=context) as resp:
                data = resp.read()
            print("Downloaded", len(data), "bytes")
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                print("Zip contents:", zf.namelist())
                zf.extractall(RAW)
            return RAW
        except Exception as exc:
            last_error = exc
            print("Failed:", exc)
    raise RuntimeError(f"Could not download UCI dataset: {last_error}")


if __name__ == "__main__":
    out = download()
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".csv", ".zip"}:
            print(path.relative_to(ROOT), path.stat().st_size)
