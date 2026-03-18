import os, sys, glob, re, tempfile, gzip, shutil
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
RAW_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "GSE134809")
os.makedirs(PROCESSED_DIR, exist_ok=True)

QC_MIN_GENES   = 200
QC_MIN_CELLS   = 3
QC_MAX_MITO    = 20.0
N_HVG          = 4000
USE_CONDITIONS = ["Involved", "Uninvolved"]

# GSM -> (condition, patient) mapping
# Paired: patient numbers match across Involved/Uninvolved
SAMPLE_MAP = {
    "GSM3972009": ("Involved",   "69"),
    "GSM3972010": ("Uninvolved", "68"),   # paired with 69
    "GSM3972011": ("Involved",   "122"),
    "GSM3972012": ("Uninvolved", "123"),
    "GSM3972013": ("Involved",   "128"),
    "GSM3972014": ("Uninvolved", "129"),
    "GSM3972015": ("Uninvolved", "135"),
    "GSM3972016": ("Involved",   "138"),
    "GSM3972017": ("Involved",   "158"),
    "GSM3972018": ("Uninvolved", "159"),
    "GSM3972019": ("Uninvolved", "180"),
    "GSM3972020": ("Involved",   "181"),
    "GSM3972021": ("Uninvolved", "186"),
    "GSM3972022": ("Involved",   "187"),
    "GSM3972023": ("Uninvolved", "189"),
    "GSM3972024": ("Involved",   "190"),
    "GSM3972025": ("Uninvolved", "192"),
    "GSM3972026": ("Involved",   "193"),
    "GSM3972027": ("Uninvolved", "195"),
    "GSM3972028": ("Involved",   "196"),
    "GSM3972029": ("Uninvolved", "208"),
    "GSM3972030": ("Involved",   "209"),
    # GSM4761136-44 are PBMC — excluded
}

def load_mtx_flat(raw_dir, gsm, condition, patient):
    """
    Load CellRanger v2 MTX from flat directory.
    Files: {GSM}_{patient}_barcodes.tsv.gz, {GSM}_{patient}_genes.tsv.gz,
           {GSM}_{patient}_matrix.mtx.gz
    """
    prefix = gsm + "_" + patient + "_"
    barcodes_path = os.path.join(raw_dir, prefix + "barcodes.tsv.gz")
    genes_path    = os.path.join(raw_dir, prefix + "genes.tsv.gz")
    matrix_path   = os.path.join(raw_dir, prefix + "matrix.mtx.gz")

    for p in [barcodes_path, genes_path, matrix_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    # Write to temp dir as standard names for sc.read_10x_mtx
    tmpdir = tempfile.mkdtemp()
    try:
        shutil.copy(barcodes_path, os.path.join(tmpdir, "barcodes.tsv.gz"))
        shutil.copy(genes_path,    os.path.join(tmpdir, "genes.tsv.gz"))
        shutil.copy(matrix_path,   os.path.join(tmpdir, "matrix.mtx.gz"))
        adata = sc.read_10x_mtx(tmpdir, var_names="gene_symbols", cache=False)
    finally:
        shutil.rmtree(tmpdir)

    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_counts=1)
    sample_id = gsm + "_" + condition + "_" + patient
    adata.obs_names = [sample_id + "_" + bc for bc in adata.obs_names]
    adata.obs["sample"]    = sample_id
    adata.obs["condition"] = condition
    adata.obs["patient"]   = patient
    adata.obs["gsm"]       = gsm
    return adata

def apply_qc(adata):
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                                log1p=False, inplace=True)
    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=QC_MIN_GENES)
    sc.pp.filter_genes(adata, min_cells=QC_MIN_CELLS)
    adata = adata[adata.obs["pct_counts_mt"] < QC_MAX_MITO].copy()
    print(f"  QC: {n_before} -> {adata.n_obs} cells")
    return adata

def select_hvg(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["norm_log"] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat_v3",
                                 layer="counts", batch_key="sample", subset=False)
    print(f"  HVGs: {adata.var['highly_variable'].sum()}")
    return adata

def main():
    print("=" * 60)
    print("CROHN'S DISEASE scRNA-seq PIPELINE")
    print("Script 01: Load, QC, Condition Labelling")
    print("Dataset: GSE134809 (Ileal Involved / Uninvolved)")
    print("=" * 60)

    # Check raw files exist
    available = sorted(glob.glob(os.path.join(RAW_DIR, "GSM*_barcodes.tsv.gz")))
    if not available:
        print("ERROR: No MTX files found in", RAW_DIR)
        print("Run the download cell first.")
        sys.exit(1)

    # Parse available GSMs
    present_gsms = set()
    for f in available:
        m = re.match(r".*(GSM\d+)_\d+_barcodes\.tsv\.gz$", f)
        if m:
            present_gsms.add(m.group(1))
    print(f"\n[1/4] Found {len(present_gsms)} GSM sample sets in raw dir")

    # Filter to ileal samples only (exclude PBMCs)
    samples_to_load = [(gsm, cond, pat) for gsm, (cond, pat) in SAMPLE_MAP.items()
                       if gsm in present_gsms and cond in USE_CONDITIONS]
    print(f"  Loading {len(samples_to_load)} ileal samples")
    for cond in USE_CONDITIONS:
        n = sum(1 for _, c, _ in samples_to_load if c == cond)
        print(f"  {cond}: {n}")

    print("\n[2/4] Loading MTX samples...")
    adatas = []
    for gsm, condition, patient in samples_to_load:
        sid = gsm + "_" + condition + "_" + patient
        print(f"  {sid}...", end=" ", flush=True)
        try:
            a = load_mtx_flat(RAW_DIR, gsm, condition, patient)
            adatas.append(a)
            print(f"{a.n_obs} cells x {a.n_vars} genes")
        except Exception as e:
            print(f"ERROR: {e}")

    if not adatas:
        print("ERROR: No samples loaded.")
        sys.exit(1)

    print(f"\n  Concatenating {len(adatas)} samples...")
    adata = sc.concat(adatas, join="outer", fill_value=0)
    adata.layers["counts"] = (sp.csr_matrix(adata.X)
                               if not sp.issparse(adata.X) else adata.X.copy())
    print(f"  Combined: {adata.n_obs} cells x {adata.n_vars} genes")
    for cond in USE_CONDITIONS:
        print(f"  {cond}:", (adata.obs["condition"] == cond).sum())

    print("\n[3/4] QC filtering...")
    adata_qc = apply_qc(adata)
    adata_qc.write_h5ad(os.path.join(PROCESSED_DIR, "ibd_qc.h5ad"))
    summary = adata_qc.obs.groupby(["condition", "patient"]).size().reset_index(name="n_cells")
    summary.to_csv(os.path.join(PROCESSED_DIR, "qc_summary.csv"), index=False)
    print("  QC summary:")
    print(summary.to_string(index=False))

    print("\n[4/4] Selecting HVGs...")
    adata_qc = select_hvg(adata_qc)
    out_path = os.path.join(PROCESSED_DIR, "adata_qc.h5ad")
    adata_qc.write_h5ad(out_path)
    print("\n" + "=" * 60)
    print("Script 01 complete. ->", out_path)
    for cond in USE_CONDITIONS:
        print(f"  {cond}:", (adata_qc.obs["condition"] == cond).sum())
    print("=" * 60)

if __name__ == "__main__":
    main()
