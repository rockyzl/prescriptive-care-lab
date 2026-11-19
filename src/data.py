"""Download (hash-checked) CMS DE-SynPUF Sample 1, build the 2008 baseline cohort,
2008-only features, 2009 outcomes and a deterministic beneficiary-level split.

Outputs (data/derived/, gitignored):
  features.parquet         one row per eligible beneficiary, 2008 information only, NO outcomes
  labels_train.parquet     2009 outcomes, train split only
  labels_val.parquet       2009 outcomes, validation split only
  labels_test.parquet      2009 outcomes, test split only (read only by `run test`)
  audit.parquet            race/sex/age-group fields for subgroup audit (2008 values; not predictors)
and data/manifest.json + data/membership.json (committed).
"""
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
MANIFEST = ROOT / "data" / "manifest.json"
MEMBERSHIP = ROOT / "data" / "membership.json"

CMS = "https://www.cms.gov/research-statistics-data-and-systems/downloadable-public-use-files/synpufs/downloads/"
SOURCES = {
    "bene_2008": CMS + "de1_0_2008_beneficiary_summary_file_sample_1.zip",
    "bene_2009": CMS + "de1_0_2009_beneficiary_summary_file_sample_1.zip",
    "inpatient": CMS + "de1_0_2008_to_2010_inpatient_claims_sample_1.zip",
    "outpatient": CMS + "de1_0_2008_to_2010_outpatient_claims_sample_1.zip",
    "carrier_a": "http://downloads.cms.gov/files/DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.zip",
    "carrier_b": "http://downloads.cms.gov/files/DE1_0_2008_to_2010_Carrier_Claims_Sample_1B.zip",
    "users_document_pdf": CMS + "synpuf_dug.pdf",
    "codebook_pdf": "https://www.cms.gov/files/document/de-10-codebook.pdf-0",
}
LOCAL_NAME = {k: ("de-10-codebook.pdf" if k == "codebook_pdf" else v.rsplit("/", 1)[1]) for k, v in SOURCES.items()}

SPLIT_SALT = "prescriptive-care-lab-v1"
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)  # train, val, test
FEATURE_YEAR = 2008
OUTCOME_YEAR = 2009

CHRONIC = ["SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD", "SP_DEPRESSN",
           "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS", "SP_RA_OA", "SP_STRKETIA"]
RACE = {1: "white", 2: "black", 3: "other", 5: "hispanic"}
ED_HCPCS = {"99281", "99282", "99283", "99284", "99285"}

# Coarse ICD-9-CM chapters (3-digit numeric ranges) plus V and E supplementary codes.
ICD9_CHAPTERS = [
    ("infectious", 1, 139), ("neoplasm", 140, 239), ("endocrine_metabolic", 240, 279), ("blood", 280, 289),
    ("mental", 290, 319), ("nervous_sense", 320, 389), ("circulatory", 390, 459), ("respiratory", 460, 519),
    ("digestive", 520, 579), ("genitourinary", 580, 629), ("pregnancy", 630, 679), ("skin", 680, 709),
    ("musculoskeletal", 710, 739), ("congenital", 740, 759), ("perinatal", 760, 779), ("symptoms_illdefined", 780, 799),
    ("injury_poisoning", 800, 999),
]
CHAPTER_NAMES = [c[0] for c in ICD9_CHAPTERS] + ["v_codes", "e_codes"]

# Fields that carry outcome-year (2009+) information or are identifiers/geography; never predictors.
FORBIDDEN_PREDICTORS = {
    "DESYNPUF_ID", "BENE_DEATH_DT", "y_ip", "cost_2009", "high_cost", "n_adm_2009", "death_2009",
    "parta_months_2009", "SP_STATE_CODE", "BENE_COUNTY_CD", "race", "race_black", "race_other", "race_hispanic",
}


# ----------------------------------------------------------------------------- provenance
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    old = json.loads(MANIFEST.read_text()).get("files", {}) if MANIFEST.exists() else {}
    files = {}
    for key, url in SOURCES.items():
        path = RAW / LOCAL_NAME[key]
        if not path.exists():
            with requests.get(url, stream=True, timeout=900, headers={"User-Agent": "research-script"}) as r:
                r.raise_for_status()
                final_url = r.url
                with open(path, "wb") as f:
                    for b in r.iter_content(1 << 20):
                        f.write(b)
        else:
            final_url = old.get(key, {}).get("final_url", url)
        digest = sha256_file(path)
        if key in old and old[key]["sha256"] != digest:
            raise ValueError(f"SHA-256 mismatch for {key}: {digest} != {old[key]['sha256']}")
        files[key] = {"url": url, "final_url": final_url, "local": f"data/raw/{path.name}", "bytes": path.stat().st_size,
                      "sha256": digest, "retrieved": old.get(key, {}).get("retrieved", datetime.now(timezone.utc).date().isoformat())}
    return files


def digest(path):
    return sha256_file(path)


# ----------------------------------------------------------------------------- split
def split_value(bene_id, salt=SPLIT_SALT):
    """Deterministic uniform [0,1) value from the beneficiary ID only (no outcomes)."""
    return int(hashlib.sha256(f"{bene_id}|{salt}".encode()).hexdigest()[:15], 16) / 16**15


def split_for(ids, fractions=SPLIT_FRACTIONS, salt=SPLIT_SALT):
    ids = list(ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Beneficiary IDs must be unique")
    a, b = fractions[0], fractions[0] + fractions[1]
    out = {}
    for i in ids:
        u = split_value(i, salt)
        out[i] = "train" if u < a else ("val" if u < b else "test")
    return out


# ----------------------------------------------------------------------------- readers
def read_zip_csv(key, **kw):
    zf = zipfile.ZipFile(RAW / LOCAL_NAME[key])
    member = zf.infolist()[0]
    kw["dtype"] = {"DESYNPUF_ID": str, **kw.get("dtype", {})}
    return pd.read_csv(zf.open(member), **kw)


def icd9_chapter(code):
    """Map an undotted ICD-9-CM diagnosis code to a coarse chapter name (None if unparseable)."""
    if not isinstance(code, str) or not code:
        return None
    c = code.strip().upper()
    if c.startswith("V"):
        return "v_codes"
    if c.startswith("E"):
        return "e_codes"
    if len(c) < 3 or not c[:3].isdigit():
        return None
    n = int(c[:3])
    for name, lo, hi in ICD9_CHAPTERS:
        if lo <= n <= hi:
            return name
    return None


_CHAPTER_CACHE = {}


def chapters_of(series):
    uniq = pd.unique(series.dropna())
    for u in uniq:
        if u not in _CHAPTER_CACHE:
            _CHAPTER_CACHE[u] = icd9_chapter(u)
    return series.map(_CHAPTER_CACHE)


def in_year(dates, year):
    d = pd.to_numeric(dates, errors="coerce")
    return (d // 10000) == year


def feature_window(df, year=FEATURE_YEAR):
    """Claims fully inside the feature year: both from- and thru-dates in `year`."""
    return df[in_year(df.CLM_FROM_DT, year) & in_year(df.CLM_THRU_DT, year)]


def admission_year(ip):
    admit = pd.to_numeric(ip.CLM_ADMSN_DT, errors="coerce").fillna(pd.to_numeric(ip.CLM_FROM_DT, errors="coerce"))
    return admit // 10000


def admissions_in_year(ip, year=OUTCOME_YEAR):
    """Distinct inpatient claims per beneficiary whose ADMISSION date falls in `year`."""
    ip_y = ip[admission_year(ip) == year].drop_duplicates(["DESYNPUF_ID", "CLM_ID"])
    return ip_y.groupby("DESYNPUF_ID").size()


def claim_block_features(df, prefix, dx_cols, pay_cols, hcpcs_cols=(), day_col=None):
    """Per-beneficiary aggregates from claims ALREADY restricted to the feature window.
    Returns (numeric aggregates, long (id, chapter) pairs)."""
    df = df.drop_duplicates(["DESYNPUF_ID", "CLM_ID"])
    g = df.groupby("DESYNPUF_ID")
    agg = pd.DataFrame({f"{prefix}_claims": g.size()})
    agg[f"{prefix}_paid"] = df[list(pay_cols)].apply(pd.to_numeric, errors="coerce").fillna(0).sum(1).groupby(df.DESYNPUF_ID).sum()
    if day_col:
        agg[f"{prefix}_days"] = pd.to_numeric(df[day_col], errors="coerce").fillna(0).groupby(df.DESYNPUF_ID).sum()
    if hcpcs_cols:
        ed = df[list(hcpcs_cols)].isin(ED_HCPCS).any(axis=1)
        agg[f"{prefix}_ed_claims"] = ed.groupby(df.DESYNPUF_ID).sum()
    long = df.melt(id_vars="DESYNPUF_ID", value_vars=list(dx_cols), value_name="dx").dropna(subset=["dx"])
    long = long[["DESYNPUF_ID", "dx"]].drop_duplicates()
    long["dx3"] = long.dx.str[:3]
    long["chapter"] = chapters_of(long.dx)
    return agg, long[["DESYNPUF_ID", "dx3", "chapter"]].drop_duplicates()


def carrier_features(key, cohort, chunksize=200_000, year=FEATURE_YEAR):
    dx = [f"ICD9_DGNS_CD_{i}" for i in range(1, 9)]
    pay = [f"LINE_NCH_PMT_AMT_{i}" for i in range(1, 14)]
    hc = [f"HCPCS_CD_{i}" for i in range(1, 14)]
    cols = ["DESYNPUF_ID", "CLM_ID", "CLM_FROM_DT", "CLM_THRU_DT"] + dx + pay + hc
    dtypes = {c: str for c in ["DESYNPUF_ID", "CLM_ID"] + dx + hc}
    aggs, longs, n_rows = [], [], 0
    zf = zipfile.ZipFile(RAW / LOCAL_NAME[key])
    for chunk in pd.read_csv(zf.open(zf.infolist()[0]), usecols=cols, dtype=dtypes, chunksize=chunksize):
        n_rows += len(chunk)
        chunk = feature_window(chunk[chunk.DESYNPUF_ID.isin(cohort)], year)
        if chunk.empty:
            continue
        a, l = claim_block_features(chunk, "car", dx, pay, hc)
        aggs.append(a)
        longs.append(l)
    agg = pd.concat(aggs).groupby(level=0).sum()
    return agg, pd.concat(longs).drop_duplicates(), n_rows


# ----------------------------------------------------------------------------- build
def recode_bene(b):
    out = pd.DataFrame(index=b.DESYNPUF_ID.values)
    out.index.name = "DESYNPUF_ID"
    birth = pd.to_datetime(b.BENE_BIRTH_DT.astype(str), format="%Y%m%d")
    out["age"] = ((pd.Timestamp(f"{FEATURE_YEAR}-12-31") - birth).dt.days / 365.25).values
    out["female"] = (b.BENE_SEX_IDENT_CD == 2).astype(int).values
    out["esrd"] = (b.BENE_ESRD_IND.astype(str) == "Y").astype(int).values
    for c in CHRONIC:
        out["cc_" + c[3:].lower()] = (b[c] == 1).astype(int).values
    out["n_chronic"] = out[[c for c in out if c.startswith("cc_")]].sum(1)
    for c in ["MEDREIMB_IP", "BENRES_IP", "PPPYMT_IP", "MEDREIMB_OP", "BENRES_OP", "PPPYMT_OP", "MEDREIMB_CAR", "BENRES_CAR", "PPPYMT_CAR"]:
        out["bene08_" + c.lower()] = b[c].astype(float).values
    out["bene08_medreimb_total"] = out[["bene08_medreimb_ip", "bene08_medreimb_op", "bene08_medreimb_car"]].sum(1)
    return out


def eligibility(b8, b9):
    steps = [("2008 Beneficiary Summary File, Sample 1", len(b8))]
    m = b8.BENE_DEATH_DT.isna()
    steps.append(("alive at end of 2008 (no 2008 death date)", int(m.sum())))
    m &= b8.DESYNPUF_ID.isin(b9.DESYNPUF_ID)
    steps.append(("also present in 2009 Beneficiary Summary File", int(m.sum())))
    m &= (b8.BENE_HI_CVRAGE_TOT_MONS == 12) & (b8.BENE_SMI_CVRAGE_TOT_MONS == 12)
    steps.append(("12 months Part A and 12 months Part B in 2008", int(m.sum())))
    m &= b8.BENE_HMO_CVRAGE_TOT_MONS == 0
    steps.append(("0 HMO months in 2008 (fee-for-service baseline)", int(m.sum())))
    return m, steps


def build():
    t0 = datetime.now(timezone.utc)
    files = download()
    b8 = read_zip_csv("bene_2008")
    b9 = read_zip_csv("bene_2009")
    if not b8.DESYNPUF_ID.is_unique or not b9.DESYNPUF_ID.is_unique:
        raise ValueError("Duplicate beneficiaries in a summary file")
    elig, steps = eligibility(b8, b9)
    b8c = b8[elig]
    cohort = set(b8c.DESYNPUF_ID)
    feats = recode_bene(b8c)

    # ---- inpatient: 2008 features, 2009 outcome
    ip_dx = [f"ICD9_DGNS_CD_{i}" for i in range(1, 11)]
    ip = read_zip_csv("inpatient", dtype={"DESYNPUF_ID": str, "CLM_ID": str, **{c: str for c in ip_dx + ["ADMTNG_ICD9_DGNS_CD"]}},
                      usecols=["DESYNPUF_ID", "CLM_ID", "SEGMENT", "CLM_FROM_DT", "CLM_THRU_DT", "CLM_ADMSN_DT", "CLM_PMT_AMT", "CLM_UTLZTN_DAY_CNT"] + ip_dx)
    ip = ip[ip.DESYNPUF_ID.isin(cohort)]
    ip_counts = {"cohort_ip_rows": len(ip), "segment_2_rows": int((ip.SEGMENT == 2).sum())}
    ip08 = feature_window(ip)
    ip_agg, ip_long = claim_block_features(ip08, "ip", ip_dx, ["CLM_PMT_AMT"], day_col="CLM_UTLZTN_DAY_CNT")
    admit = admission_year(ip) * 10000
    thru_year = pd.to_numeric(ip.CLM_THRU_DT, errors="coerce") // 10000
    ip_counts["admitted_2007_rows"] = int((admit // 10000 == 2007).sum())
    ip_counts["admitted_2008_thru_2009_rows"] = int(((admit // 10000 == 2008) & (thru_year == 2009)).sum())
    ip_counts["note"] = ("Feature claims must have from- and thru-dates in 2008, so stays spanning 2007/2008 or 2008/2009 are not "
                         "claim features (their payments are still inside the 2008 Beneficiary Summary totals). Outcome = admission "
                         "date (CLM_ADMSN_DT, else CLM_FROM_DT) in 2009; stays admitted in 2008 are never outcomes.")
    n_adm_2009 = admissions_in_year(ip, OUTCOME_YEAR)

    # ---- outpatient
    op_dx = [f"ICD9_DGNS_CD_{i}" for i in range(1, 11)]
    op_hc = [f"HCPCS_CD_{i}" for i in range(1, 46)]
    op = read_zip_csv("outpatient", usecols=["DESYNPUF_ID", "CLM_ID", "CLM_FROM_DT", "CLM_THRU_DT", "CLM_PMT_AMT"] + op_dx + op_hc,
                      dtype={"DESYNPUF_ID": str, "CLM_ID": str, **{c: str for c in op_dx + op_hc}})
    op = feature_window(op[op.DESYNPUF_ID.isin(cohort)])
    op_agg, op_long = claim_block_features(op, "op", op_dx, ["CLM_PMT_AMT"], hcpcs_cols=op_hc)
    del op

    # ---- carrier (streamed; ~1.2 GB per CSV)
    car_aggs, car_longs, car_rows = [], [], {}
    for key in ["carrier_a", "carrier_b"]:
        a, l, n = carrier_features(key, cohort)
        car_aggs.append(a)
        car_longs.append(l)
        car_rows[key] = n
    car_agg = pd.concat(car_aggs).groupby(level=0).sum()

    for agg in [ip_agg, op_agg, car_agg]:
        feats = feats.join(agg, how="left")
    long = pd.concat([ip_long, op_long] + car_longs).drop_duplicates()
    chap = pd.crosstab(long.DESYNPUF_ID, long.chapter).clip(upper=1)
    chap = chap.reindex(columns=CHAPTER_NAMES, fill_value=0).add_prefix("dx_")
    feats = feats.join(chap, how="left")
    feats["n_distinct_dx3"] = long.drop_duplicates(["DESYNPUF_ID", "dx3"]).groupby("DESYNPUF_ID").size()
    feats = feats.fillna(0)
    feats["ed_claims"] = feats.pop("op_ed_claims") + feats.pop("car_ed_claims")
    feats["any_ip_2008"] = (feats.ip_claims > 0).astype(int)

    # ---- outcomes (2009) and audit fields
    b9i = b9.set_index("DESYNPUF_ID").reindex(feats.index)
    labels = pd.DataFrame(index=feats.index)
    labels["n_adm_2009"] = n_adm_2009.reindex(feats.index).fillna(0).astype(int)
    labels["y_ip"] = (labels.n_adm_2009 > 0).astype(int)
    labels["cost_2009"] = b9i[["MEDREIMB_IP", "MEDREIMB_OP", "MEDREIMB_CAR"]].sum(1).astype(float)
    labels["death_2009"] = b9i.BENE_DEATH_DT.notna().astype(int)
    labels["parta_months_2009"] = b9i.BENE_HI_CVRAGE_TOT_MONS.astype(int)

    audit = pd.DataFrame(index=feats.index)
    b8i = b8c.set_index("DESYNPUF_ID")
    audit["race"] = b8i.BENE_RACE_CD.map(RACE).reindex(feats.index)
    audit["sex"] = np.where(feats.female == 1, "female", "male")
    audit["age_group"] = pd.cut(feats.age, [0, 65, 75, 85, 200], right=False, labels=["<65", "65-74", "75-84", "85+"]).astype(str)

    membership = split_for(feats.index)
    split = pd.Series(membership).reindex(feats.index)
    feats.insert(0, "split", split.values)
    audit["split"] = split.values

    DERIVED.mkdir(parents=True, exist_ok=True)
    feats.reset_index().to_parquet(DERIVED / "features.parquet", index=False)
    audit.reset_index().to_parquet(DERIVED / "audit.parquet", index=False)
    for part in ["train", "val", "test"]:
        labels[split == part].reset_index().to_parquet(DERIVED / f"labels_{part}.parquet", index=False)

    MEMBERSHIP.write_text(json.dumps(dict(sorted(membership.items())), indent=0))
    counts = split.value_counts().to_dict()
    manifest = {
        "created_at": t0.isoformat(),
        "dataset": "CMS 2008-2010 Data Entrepreneurs' Synthetic Public Use File (DE-SynPUF), DE1.0 Sample 1",
        "synthetic": True,
        "landing_pages": {
            "sample_1": "https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf/de10-sample-1",
            "de_synpuf": "https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf",
            "synpufs": "https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files",
        },
        "terms": {
            "summary": "Free public-use download from CMS; no data use agreement required. CMS describes the DE-SynPUF as created 'with the goal of providing a realistic set of claims data in the public domain'. The SynPUFs page asks users to read the 'CMS Data Disclaimer - User Agreement - Public Use Data'; as retrieved on 2026-11-19 the only file in that page's Downloads section is the DE 1.0 Data Users Document (synpuf_dug.pdf, hashed below).",
            "inference_warning_quote": "the DE-SynPUF has very limited inferential research utility because of the synthetic process used to generate the data. That is, analyses using the DE-SynPUF to draw inferences about Medicare beneficiaries, providers, or the Medicare program will be misleading and often incorrect. (DE-SynPUF User Manual, synpuf_dug.pdf, section 6)",
            "contains_phi": False,
            "cpt_note": "HCPCS Level I (CPT) codes are AMA-copyrighted; this project only matches five numeric ED evaluation-and-management codes (99281-99285) and redistributes no descriptors.",
        },
        "page_quirks": ["On the Sample 1 landing page (retrieved 2026-11-19) the '2010 Beneficiary Summary File' link points to Sample 20's 2010 file. 2010 data are not used in this project."],
        "files": files,
        "files_not_used": ["2010 Beneficiary Summary File", "Prescription Drug Events (2008-2010)"],
        "cohort_steps": [{"step": s, "n": n} for s, n in steps],
        "cohort_n": int(len(feats)),
        "split": {part: {"n": int(counts.get(part, 0)), "y_ip_events": int(labels.y_ip[split == part].sum())} for part in ["train", "val", "test"]},
        "split_rule": f"u = int(SHA256(DESYNPUF_ID|{SPLIT_SALT})[:15], 16)/16^15; train u<0.6, val u<0.8, else test. Uses the ID only.",
        "inpatient_claim_notes": ip_counts,
        "carrier_rows_read": car_rows,
        "outcome_year_parta_zero_months": int((labels.parta_months_2009 == 0).sum()),
        "deaths_2009": int(labels.death_2009.sum()),
        "feature_columns": [c for c in feats.columns if c != "split"],
        "n_features": len(feats.columns) - 1,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ["cohort_steps", "split", "inpatient_claim_notes", "outcome_year_parta_zero_months", "deaths_2009", "n_features"]}, indent=1))


if __name__ == "__main__":
    build()
