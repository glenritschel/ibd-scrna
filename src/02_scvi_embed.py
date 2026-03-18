import os, sys
import numpy as np
import pandas as pd
import scanpy as sc

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

SCVI_PARAMS        = {"n_latent": 30, "n_layers": 2, "n_hidden": 128}
SCVI_TRAIN_PARAMS  = {"max_epochs": 400, "early_stopping": False}
N_NEIGHBORS        = 15
RANDOM_SEED        = 0
LEIDEN_RESOLUTIONS = [0.5, 0.8, 1.2]

# Marker panels for intestinal cell types
CELL_TYPE_MARKERS = {
    "colonocyte_absorptive": ["EPCAM","SLC26A3","CA1","CA2","KRT20","CDX2","FABP1"],
    "colonocyte_best4":      ["BEST4","OTOP2","SPIB","CFTR","NOTCH2"],
    "goblet":                ["MUC2","TFF3","CLCA1","ZG16","FCGBP","SPDEF"],
    "stem_progenitor":       ["LGR5","OLFM4","ASCL2","SMOC2","SOX9","RGMB"],
    "paneth":                ["DEFA5","DEFA6","LYZ","PRSS2","ITLN2"],
    "enteroendocrine":       ["CHGA","CHGB","SYP","GCG","CCK","SST"],
    "fibroblast":            ["DCN","LUM","COL1A1","PDGFRA","FAP","THY1"],
    "inf_fibroblast":        ["IL13RA2","IL11","WNT5A","CXCL14","PDPN","TNFRSF11B"],
    "smooth_muscle":         ["ACTA2","MYH11","CNN1","DES","TAGLN"],
    "t_cell":                ["CD3D","CD3E","CD8A","CD4","TRAC","IL7R"],
    "b_cell":                ["CD79A","MS4A1","CD19","MZB1","JCHAIN","IGHG1"],
    "myeloid":               ["CD68","LYZ","S100A9","CD14","FCGR3A","IL1B"],
    "endothelial":           ["VWF","PECAM1","CDH5","CLDN5","ACKR1"],
    "mast_cell":             ["TPSAB1","TPSB2","CPA3","KIT"],
}

def set_seeds(seed=0):
    import torch, random
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)

def train_scvi(adata):
    import scvi
    scvi.settings.seed = RANDOM_SEED
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="sample")
    model = scvi.model.SCVI(adata, **SCVI_PARAMS)
    print(f"  Training scVI on {adata.n_obs} cells x {adata.var['highly_variable'].sum()} HVGs...")
    model.train(**SCVI_TRAIN_PARAMS, accelerator="auto")
    final_loss = float(np.array(model.history["train_loss_epoch"].values[-1]).flat[0])
    print(f"  Training complete. Final loss: {final_loss:.2f}")
    adata.obsm["X_scVI"] = model.get_latent_representation()
    print(f"  Latent shape: {adata.obsm['X_scVI'].shape}")
    return adata, model

def score_resolution(adata, resolution_results):
    rows = []
    for res, info in resolution_results.items():
        key = info["key"]
        best_clusters = {}
        for ct, markers in CELL_TYPE_MARKERS.items():
            present = [m for m in markers if m in adata.var_names]
            if not present: continue
            cluster_means = {}
            for cl in adata.obs[key].unique():
                mask = adata.obs[key] == cl
                e = adata[mask, present].X
                if hasattr(e, "toarray"): e = e.toarray()
                cluster_means[cl] = float(e.mean())
            best_clusters[ct] = max(cluster_means, key=cluster_means.get)
        n_distinct = len(set(best_clusters.values()))
        n_resolved = len(best_clusters)
        rows.append({
            "resolution": res,
            "n_clusters": info["n_clusters"],
            "n_celltypes_resolved": n_resolved,
            "n_distinct_best_clusters": n_distinct,
            "separation_score": n_distinct / max(n_resolved, 1),
        })
    df = pd.DataFrame(rows).sort_values("separation_score", ascending=False)
    best = df.iloc[0]
    recommended = best["resolution"]
    print("\n  Resolution comparison:")
    print(df.to_string(index=False))
    print(f"\n  Recommended: {recommended} ({int(best['n_clusters'])} clusters, "
          f"score: {float(best['separation_score']):.2f})")
    return df, recommended

def main():
    print("=" * 60)
    print("CROHN'S DISEASE scRNA-seq PIPELINE")
    print("Script 02: scVI + Leiden Clustering")
    print("=" * 60)
    in_path = os.path.join(PROCESSED_DIR, "adata_qc.h5ad")
    if not os.path.exists(in_path):
        print("ERROR:", in_path, "not found. Run 01_load_qc.py first.")
        sys.exit(1)

    print("\n[1/5] Loading QC object...")
    adata = sc.read_h5ad(in_path)
    print(f"  Loaded: {adata.n_obs} cells x {adata.n_vars} genes")
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    print(f"  HVG subset: {adata_hvg.n_obs} x {adata_hvg.n_vars}")

    print("\n[2/5] Training scVI...")
    set_seeds(RANDOM_SEED)
    adata_hvg, model = train_scvi(adata_hvg)

    print("\n[3/5] Building neighbour graph and UMAP...")
    sc.pp.neighbors(adata_hvg, use_rep="X_scVI", n_neighbors=N_NEIGHBORS)
    sc.tl.umap(adata_hvg)

    print("\n[4/5] Running Leiden at multiple resolutions...")
    res_results = {}
    for res in LEIDEN_RESOLUTIONS:
        key = f"leiden_{res}"
        sc.tl.leiden(adata_hvg, resolution=res, key_added=key,
                     random_state=RANDOM_SEED, flavor="igraph",
                     n_iterations=2, directed=False)
        n = adata_hvg.obs[key].nunique()
        print(f"  Resolution {res}: {n} clusters")
        res_results[res] = {"key": key, "n_clusters": n}

    print("\n[5/5] Selecting best resolution...")
    res_metrics, recommended = score_resolution(adata_hvg, res_results)
    adata_hvg.obs["leiden"] = adata_hvg.obs[f"leiden_{recommended}"].copy()
    adata_hvg.uns["recommended_leiden_resolution"] = recommended
    adata_hvg.uns["pro_ibd_clusters"] = []

    out_path = os.path.join(PROCESSED_DIR, "adata_scvi.h5ad")
    adata_hvg.write_h5ad(out_path)
    res_metrics.to_csv(os.path.join(PROCESSED_DIR, "resolution_metrics.csv"), index=False)
    cond_dist = adata_hvg.obs.groupby(["leiden", "condition"]).size().unstack(fill_value=0)
    cond_dist.to_csv(os.path.join(PROCESSED_DIR, "cluster_condition_distribution.csv"))

    print("\n" + "=" * 60)
    print("Script 02 complete. ->", out_path)
    print(f"  Final clusters: {adata_hvg.obs['leiden'].nunique()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
