import os, sys
import numpy as np
import pandas as pd
import scanpy as sc

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

CELL_TYPE_MARKERS = {
    "colonocyte_absorptive": ["EPCAM","SLC26A3","CA1","CA2","KRT20","CDX2","FABP1"],
    "colonocyte_best4":      ["BEST4","OTOP2","SPIB","CFTR","NOTCH2"],
    "goblet":                ["MUC2","TFF3","CLCA1","ZG16","FCGBP","SPDEF"],
    "stem_progenitor":       ["LGR5","OLFM4","ASCL2","SMOC2","SOX9"],
    "paneth":                ["DEFA5","DEFA6","LYZ","PRSS2","ITLN2"],
    "enteroendocrine":       ["CHGA","CHGB","SYP","GCG","CCK"],
    "fibroblast":            ["DCN","LUM","COL1A1","PDGFRA","FAP"],
    "inf_fibroblast":        ["IL13RA2","IL11","WNT5A","CXCL14","PDPN"],
    "smooth_muscle":         ["ACTA2","MYH11","CNN1","DES","TAGLN"],
    "t_cell":                ["CD3D","CD3E","CD8A","IL7R","TRAC"],
    "b_cell":                ["CD79A","MS4A1","MZB1","XBP1","JCHAIN"],
    "myeloid":               ["CD68","LYZ","S100A9","CD14","IL1B"],
    "endothelial":           ["VWF","PECAM1","CDH5","CLDN5"],
    "mast_cell":             ["TPSAB1","TPSB2","CPA3","KIT"],
}
CONFIDENCE_THRESHOLD = 1.2

def score_clusters(adata, marker_dict, cluster_key="leiden"):
    clusters = sorted(adata.obs[cluster_key].unique(), key=lambda x: int(x))
    scores = {ct: [] for ct in marker_dict}
    for cl in clusters:
        mask = adata.obs[cluster_key] == cl
        for ct, markers in marker_dict.items():
            present = [m for m in markers if m in adata.var_names]
            if not present:
                scores[ct].append(0.0); continue
            e = adata[mask, present].X
            if hasattr(e, "toarray"): e = e.toarray()
            scores[ct].append(float(e.mean()))
    df = pd.DataFrame(scores, index=clusters)
    df.index.name = "cluster"
    return df

def assign_annotations(score_df):
    rows = []
    for cluster in score_df.index:
        row = score_df.loc[cluster]
        best = row.idxmax()
        best_score = row.max()
        sorted_s = row.sort_values(ascending=False)
        runner_up = sorted_s.iloc[1] if len(sorted_s) > 1 else 0.0
        if best_score == 0.0:
            label, conf = "unknown", "low"
        elif runner_up == 0.0 or best_score >= CONFIDENCE_THRESHOLD * runner_up:
            label, conf = best, "high"
        else:
            label, conf = best + "_mixed", "low"
        rows.append({"cluster": cluster, "annotation": label, "confidence": conf,
                     "best_score": round(best_score, 4), "runner_up_score": round(runner_up, 4)})
    return pd.DataFrame(rows)

def main():
    print("=" * 60)
    print("CROHN'S DISEASE scRNA-seq PIPELINE")
    print("Script 03: Cell Type Annotation")
    print("=" * 60)
    in_path = os.path.join(PROCESSED_DIR, "adata_scvi.h5ad")
    if not os.path.exists(in_path):
        print("ERROR:", in_path, "not found. Run 02_scvi_embed.py first.")
        sys.exit(1)

    print("\n[1/4] Loading scVI object...")
    adata = sc.read_h5ad(in_path)
    print(f"  Loaded: {adata.n_obs} cells, {adata.obs['leiden'].nunique()} clusters")
    if "norm_log" not in adata.layers:
        sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)

    print("\n[2/4] Scoring clusters against intestinal cell type markers...")
    score_df = score_clusters(adata, CELL_TYPE_MARKERS)

    print("\n[3/4] Assigning annotations...")
    ann_df = assign_annotations(score_df)
    counts = adata.obs["leiden"].value_counts().rename("n_cells")
    summary = ann_df.set_index("cluster").join(counts).sort_values("n_cells", ascending=False)
    print("\n  Cluster annotations:")
    print("  " + "-" * 70)
    for cl, row in summary.iterrows():
        print(f"  Cluster {cl} | {row['annotation']} | {row['confidence']} | {int(row['n_cells'])} cells")
    epi_clusters = ann_df[ann_df["annotation"].str.contains("colonocyte|goblet|stem|paneth|enteroendocrine", na=False)]
    print(f"\n  Epithelial clusters: {len(epi_clusters)}")
    inf_fb = ann_df[ann_df["annotation"].str.contains("inf_fibroblast", na=False)]
    print(f"  Inflammatory fibroblast clusters: {len(inf_fb)}")

    print("\n[4/4] Saving...")
    adata.obs["cell_type"] = adata.obs["leiden"].map(
        dict(zip(ann_df["cluster"].astype(str), ann_df["annotation"])))
    adata.obs["annotation_confidence"] = adata.obs["leiden"].map(
        dict(zip(ann_df["cluster"].astype(str), ann_df["confidence"])))
    out_path = os.path.join(PROCESSED_DIR, "adata_annotated.h5ad")
    adata.write_h5ad(out_path)
    ann_df.to_csv(os.path.join(PROCESSED_DIR, "cluster_annotations.csv"), index=False)
    score_df.to_csv(os.path.join(PROCESSED_DIR, "cluster_marker_scores.csv"))
    print("\n" + "=" * 60)
    print("Script 03 complete. ->", out_path)
    print("=" * 60)

if __name__ == "__main__":
    main()
