# Third-party data and attribution

This repository contains **code, fitted model files and aggregate results only**. Raw CMS files and derived per-beneficiary tables are not redistributed (see `.gitignore`). `python -m src.data` downloads them and checks them against the SHA-256 hashes in `data/manifest.json`. `data/membership.json` lists synthetic beneficiary IDs (`DESYNPUF_ID`) and their split. CMS states that these IDs "carry no information about the patient or any patient records".

| Component | Source | Terms | Use here |
|---|---|---|---|
| Beneficiary Summary 2008 and 2009; Inpatient, Outpatient and Carrier claims 2008–2010 (DE1.0 Sample 1) | Centers for Medicare & Medicaid Services, *CMS 2008–2010 Data Entrepreneurs' Synthetic Public Use File (DE-SynPUF)*, cms.gov | Free public-use files, no data use agreement. CMS describes them as "a realistic set of claims data in the public domain". Synthetic, no PHI. | Downloaded at run time; not redistributed. |
| DE-SynPUF User Manual and Codebook (PDF) | cms.gov | Public | Read for variable definitions; hashed in the manifest. |
| HCPCS / CPT codes 99281–99285 | AMA CPT (Level I HCPCS) | AMA copyright on descriptors | Five numeric codes are matched to count ED visits. No descriptors are reproduced. |

**CMS caveat (DE-SynPUF User Manual):** "analyses using the DE-SynPUF to draw inferences about Medicare beneficiaries, providers, or the Medicare program will be misleading and often incorrect."

Software dependencies (LightGBM, scikit-learn, SHAP, PuLP with the bundled COIN-OR CBC solver, pandas, matplotlib) keep their own licenses.

Research and education only. This is not a clinical tool and has no clinical validation.
