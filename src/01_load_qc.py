"""
01_load_qc.py  —  IBD scRNA-seq  (GSE134809, Smillie et al.)
Flat MTX layout: all files in one directory, prefixed by GSM ID.
Pattern: {GSM}_{patient}_{barcodes|genes|matrix}.tsv.gz / .mtx.gz
"""

import os, gc, re, shutil, tempfile
import scanpy as sc
import anndata as ad
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DRIVE_BASE  = "/content/drive/MyDrive/Ritschel_Research/ibd_scrna_output"
RAW_DIR     = os.path.join(DRIVE_BASE, "raw")
OUT_DIR     = DRIVE_BASE
os.makedirs(OUT_DIR, exist_ok=True)

# ── condition map  (GSM → condition, patient_id) ───────────────────────────
# GSE134809: ileal biopsies.  Involved = active Crohn's, Uninvolved = paired
# GSM3972009-030 = original 22 samples; GSM4761136-144 = 9 additional
CONDITION_MAP = {
    # Involved
    "GSM3972009": ("Involved",   "69"),
    "GSM3972011": ("Involved",  "122"),
    "GSM3972013": ("Involved",  "128"),
    "GSM3972016": ("Involved",  "138"),
    "GSM3972017": ("Involved",  "158"),
    "GSM3972020": ("Involved",  "181"),
    "GSM3972022": ("Involved",  "187"),
    "GSM3972024": ("Involved",  "190"),
    "GSM3972026": ("Involved",  "193"),
    "GSM3972028": ("Involved",  "196"),
    "GSM3972030": ("Involved",  "209"),
    # Uninvolved
    "GSM3972010": ("Uninvolved", "68"),
    "GSM3972012": ("Uninvolved","123"),
    "GSM3972014": ("Uninvolved","129"),
    "GSM3972015": ("Uninvolved","135"),
    "GSM3972018": ("Uninvolved","159"),
    "GSM3972019": ("Uninvolved","180"),
    "GSM3972021": ("Uninvolved","186"),
    "GSM3972023": ("Uninvolved","189"),
    "GSM3972025": ("Uninvolved","192"),
    "GSM3972027": ("Uninvolved","195"),
    "GSM3972029": ("Uninvolved","208"),
    # Additional samples — condition TBD; defaulting to label from GSE metadata
    # Add/correct these once you confirm their condition from the GEO page
    "GSM4761136": ("Uninvolved", "67"),
    "GSM4761137": ("Involved",  "126"),
    "GSM4761138": ("Uninvolved","127"),
    "GSM4761139": ("Uninvolved","134"),
    "GSM4761140": ("Uninvolved","157"),
    "GSM4761141": ("Uninvolved","179"),
    "GSM4761142": ("Uninvolved","185"),
    "GSM4761143": ("Uninvolved","191"),
    "GSM4761144": ("Uninvolved","194"),
}

# ── discover samples from flat directory ──────────────────────────────────
def discover_samples(raw_dir):
    """Return dict: gsm_id → {barcodes, genes, matrix} absolute paths."""
    samples = {}
    for fname in os.listdir(raw_dir):
        m = re.match(r'^(GSM\d+)_\d+_(barcodes|genes|matrix)\.(tsv|mtx)\.gz$', fname)
        if not m:
            continue
        gsm, ftype = m.group(1), m.group(2)
        if gsm not in samples:
            samples[gsm] = {}
        samples[gsm][ftype] = os.path.join(raw_dir, fname)
    return samples

# ── load one sample via temp dir ──────────────────────────────────────────
def load_sample(gsm, paths, condition, patient):
    """Copy flat files into a temp dir with standard names, load with scanpy."""
    tmp = tempfile.mkdtemp()
    try:
        shutil.copy(paths["barcodes"], os.path.join(tmp, "barcodes.tsv.gz"))
        shutil.copy(paths["matrix"],   os.path.join(tmp, "matrix.mtx.gz"))
        # CellRanger v2 uses genes.tsv — rename to features.tsv for scanpy
        shutil.copy(paths["genes"],    os.path.join(tmp, "features.tsv.gz"))
        adata = sc.read_10x_mtx(tmp, var_names="gene_symbols", cache=False)
        adata.obs["condition"] = condition
        adata.obs["patient"]   = patient
        adata.obs["sample"]    = gsm
        adata.obs_names        = [f"{gsm}_{bc}" for bc in adata.obs_names]
        return adata
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# ── main ──────────────────────────────────────────────────────────────────
print("Discovering samples in flat raw directory...")
all_samples = discover_samples(RAW_DIR)
print(f"  Found {len(all_samples)} GSM IDs: {sorted(all_samples.keys())}")

# Check for any GSMs not in condition map
unknown = [g for g in all_samples if g not in CONDITION_MAP]
if unknown:
    print(f"  WARNING: No condition mapping for: {unknown} — skipping these samples")

adatas_all = []
for cond in ["Involved", "Uninvolved"]:
    print(f"\nLoading {cond} samples...")
    adatas = []
    for gsm, paths in sorted(all_samples.items()):
        if gsm not in CONDITION_MAP:
            continue
        c, patient = CONDITION_MAP[gsm]
        if c != cond:
            continue
        if not all(k in paths for k in ["barcodes", "genes", "matrix"]):
            print(f"  {gsm}... SKIP (missing files: {set(paths.keys())})")
            continue
        try:
            adata = load_sample(gsm, paths, cond, patient)
            print(f"  {gsm} (patient {patient})... {adata.n_obs} cells")
            adatas.append(adata)
        except Exception as e:
            print(f"  {gsm}... ERROR: {e}")

    if not adatas:
        print(f"  WARNING: No {cond} samples loaded — check paths and condition map")
        continue

    print(f"  Concatenating {cond} ({len(adatas)} samples)...")
    batch = sc.concat(adatas, join="outer", fill_value=0)
    del adatas; gc.collect()
    print(f"  {cond}: {batch.n_obs} cells")
    adatas_all.append(batch)

print("\nMerging all conditions...")
adata = sc.concat(adatas_all, join="outer", fill_value=0)
del adatas_all; gc.collect()
print(f"Combined: {adata.n_obs} cells × {adata.n_vars} genes")

# ── QC ────────────────────────────────────────────────────────────────────
print("\nRunning QC...")
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

# Filters (standard; adjust if needed after inspecting violin plots)
n_before = adata.n_obs
adata = adata[adata.obs["n_genes_by_counts"] >= 200].copy()
adata = adata[adata.obs["n_genes_by_counts"] <= 6000].copy()
adata = adata[adata.obs["pct_counts_mt"] <= 20].copy()
print(f"  After QC: {adata.n_obs} cells ({n_before - adata.n_obs} removed)")

# ── Normalise + HVG ───────────────────────────────────────────────────────
print("\nNormalising and selecting HVGs...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata.copy()

sc.pp.highly_variable_genes(adata, n_top_genes=3000, batch_key="sample", subset=True)
print(f"  HVGs selected: {adata.n_vars}")

# ── Save ──────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "01_loaded.h5ad")
adata.write_h5ad(out_path)
print(f"\nSaved: {out_path}")
print("Script 01 complete.")
