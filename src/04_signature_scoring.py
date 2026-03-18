import os, sys
import numpy as np
import pandas as pd
import scanpy as sc

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Five CD ileal-relevant gene signatures
CD_SIGNATURES = {
    "epithelial_barrier_dysfunction": [
        "CLDN1","CLDN2","CLDN4","OCLN","TJP1","MUC2","MUC5B",
        "SPDEF","KLF4","CDX2","FABP1","SLC26A3","DPP4","VIL1"
    ],
    "innate_immune_activation": [
        "S100A8","S100A9","IL1B","TNF","CXCL1","CXCL2","CXCL8",
        "IL6","IL18","NLRP3","CASP1","NOD2","RIPK2","PTGS2"
    ],
    "th1_th17_axis": [
        "IFNG","IL12A","IL12B","IL23A","IL17A","IL17F","IL22",
        "STAT1","STAT3","STAT4","TBX21","RORC","IRF1","CXCL10"
    ],
    "fibrosis_remodelling": [
        "COL1A1","COL1A2","COL3A1","FN1","ACTA2","TGFB1","TGFB2",
        "TIMP1","MMP1","MMP3","MMP9","CTGF","POSTN","PDGFRA"
    ],
    "inf_fibroblast_programme": [
        "IL13RA2","IL11","WNT5A","IL24","CXCL14","PDPN","TNFRSF11B",
        "MMP1","MMP3","CCL2","CCL7","CCL8","OSMR","FAP"
    ],
}

def score_signatures(adata, signatures, cluster_key="leiden"):
    clusters = sorted(adata.obs[cluster_key].unique(), key=lambda x: int(x))
    cluster_scores = {sig: [] for sig in signatures}
    for sig_name, genes in signatures.items():
        present = [g for g in genes if g in adata.var_names]
        missing = [g for g in genes if g not in adata.var_names]
        msg = f"  {sig_name}: {len(present)}/{len(genes)} genes found"
        if missing: msg += f" (missing: {missing[:3]}{'...' if len(missing)>3 else ''})"
        print(msg)
        if not present:
            adata.obs["score_" + sig_name] = 0.0
            cluster_scores[sig_name].extend([0.0] * len(clusters))
            continue
        e = adata[:, present].X
        if hasattr(e, "toarray"): e = e.toarray()
        cell_scores = np.array(e.mean(axis=1)).flatten()
        adata.obs["score_" + sig_name] = cell_scores
        for cl in clusters:
            mask = adata.obs[cluster_key] == cl
            cluster_scores[sig_name].append(float(cell_scores[mask].mean()))
    df = pd.DataFrame(cluster_scores, index=clusters)
    df.index.name = "cluster"
    return adata, df

def score_by_condition(adata, signatures):
    rows = []
    for cond in ["Involved", "Uninvolved"]:
        mask = adata.obs["condition"] == cond
        if not mask.any(): continue
        row = {"condition": cond, "n_cells": int(mask.sum())}
        for sig in signatures:
            col = "score_" + sig
            row[sig] = round(float(adata.obs.loc[mask, col].mean()), 4) \
                       if col in adata.obs.columns else 0.0
        rows.append(row)
    return pd.DataFrame(rows)

def score_by_cell_type(adata, signatures):
    rows = []
    for ct in adata.obs["cell_type"].unique():
        mask = adata.obs["cell_type"] == ct
        row = {"cell_type": ct, "n_cells": int(mask.sum())}
        for sig in signatures:
            col = "score_" + sig
            row[sig] = round(float(adata.obs.loc[mask, col].mean()), 4) \
                       if col in adata.obs.columns else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_cells", ascending=False)

def main():
    print("=" * 60)
    print("CROHN'S DISEASE scRNA-seq PIPELINE")
    print("Script 04: CD Signature Scoring")
    print("=" * 60)
    in_path = os.path.join(PROCESSED_DIR, "adata_annotated.h5ad")
    if not os.path.exists(in_path):
        print("ERROR:", in_path, "not found. Run 03_annotate_clusters.py first.")
        sys.exit(1)

    print("\n[1/5] Loading annotated object...")
    adata = sc.read_h5ad(in_path)
    print(f"  Loaded: {adata.n_obs} cells x {adata.n_vars} genes")
    if "norm_log" in adata.layers:
        adata.X = adata.layers["norm_log"]
    else:
        sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)

    print("\n[2/5] Scoring CD signatures...")
    adata, score_df = score_signatures(adata, CD_SIGNATURES)
    score_df["cd_primary_score"] = (
        score_df.get("innate_immune_activation", 0) +
        score_df.get("th1_th17_axis", 0))
    score_df["cd_composite_score"] = score_df[
        [c for c in CD_SIGNATURES if c in score_df.columns]].sum(axis=1)
    top_clusters = score_df.nlargest(3, "cd_primary_score")
    print("\n  Top 3 pro-CD clusters:")
    for cl, row in top_clusters.iterrows():
        ct = adata.obs.loc[adata.obs["leiden"] == str(cl), "cell_type"].values[0] \
             if "cell_type" in adata.obs.columns else "unknown"
        print(f"    Cluster {cl} ({ct}): innate={row.get('innate_immune_activation',0):.4f}, "
              f"th1/17={row.get('th1_th17_axis',0):.4f}")

    print("\n[3/5] Scores by condition (Involved vs Uninvolved)...")
    cond_scores = score_by_condition(adata, CD_SIGNATURES)
    print(cond_scores.round(4).to_string(index=False))

    print("\n[4/5] Scores by cell type...")
    ct_scores = score_by_cell_type(adata, CD_SIGNATURES) if "cell_type" in adata.obs.columns \
                else pd.DataFrame()
    if not ct_scores.empty:
        print(ct_scores.head(8).round(4).to_string(index=False))

    print("\n[5/5] Saving...")
    adata.uns["pro_ibd_clusters"] = top_clusters.index.tolist()
    out_path = os.path.join(PROCESSED_DIR, "adata_scored.h5ad")
    adata.write_h5ad(out_path)
    score_df.to_csv(os.path.join(PROCESSED_DIR, "signature_scores.csv"))
    cond_scores.to_csv(os.path.join(PROCESSED_DIR, "signature_scores_by_condition.csv"), index=False)
    if not ct_scores.empty:
        ct_scores.to_csv(os.path.join(PROCESSED_DIR, "signature_scores_bytype.csv"), index=False)
    print("\n" + "=" * 60)
    print("Script 04 complete. ->", out_path)
    print("=" * 60)

if __name__ == "__main__":
    main()
