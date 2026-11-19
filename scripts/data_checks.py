"""Descriptive data checks cited in the docs (synthetic-data artefacts). No modelling, no split use.

  uv run python scripts/data_checks.py   -> results/data_checks.json
Cohort base: alive at end of 2008 and present in the 2009 file (114,538), as in the exploration before the protocol.
"""
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import LOCAL_NAME, RAW, admissions_in_year, read_zip_csv  # noqa: E402


def main():
    b8, b9 = read_zip_csv("bene_2008"), read_zip_csv("bene_2009")
    base = b8[b8.BENE_DEATH_DT.isna() & b8.DESYNPUF_ID.isin(b9.DESYNPUF_ID)]
    ip = read_zip_csv("inpatient", usecols=["DESYNPUF_ID", "CLM_ID", "CLM_FROM_DT", "CLM_ADMSN_DT"], dtype={"CLM_ID": str})
    admitted = set(admissions_in_year(ip, 2009).index)
    y = base.DESYNPUF_ID.isin(admitted)
    hmo = base.BENE_HMO_CVRAGE_TOT_MONS > 0
    full = (base.BENE_HI_CVRAGE_TOT_MONS == 12) & (base.BENE_SMI_CVRAGE_TOT_MONS == 12) & ~hmo
    carrier_sizes = {}
    for key in ["carrier_a", "carrier_b"]:
        info = zipfile.ZipFile(RAW / LOCAL_NAME[key]).infolist()[0]
        carrier_sizes[key] = {"member": info.filename, "uncompressed_bytes": info.file_size}
    out = {
        "base": "alive at end of 2008 and present in 2009 Beneficiary Summary File (Sample 1)",
        "n_base": int(len(base)),
        "admission_2009_rate_any_hmo_month_2008": float(y[hmo].mean()), "n_any_hmo_month": int(hmo.sum()),
        "admission_2009_rate_no_hmo_month_2008": float(y[~hmo].mean()), "n_no_hmo_month": int((~hmo).sum()),
        "note": "In real Medicare data HMO (Medicare Advantage) months have no FFS claims; a near-identical admission rate indicates a synthetic artefact.",
        "carrier_csv_sizes": carrier_sizes,
        "cohort_check_n": int(full.sum()),
    }
    (ROOT / "results" / "data_checks.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
