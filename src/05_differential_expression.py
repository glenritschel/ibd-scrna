import os, sys
import pandas as pd
import scanpy as sc

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)
N_TOP_GENES = 150

def main():
    print("=" * 60)
    print("CROHN'S DISEASE scRNA-seq PIPELINE")
    print("Script 05: Differential Expression")
    print("Involved (inflamed) vs Uninvolved within each cluster")
    print("=" * 60)
    in_path = os.path.join(PROCESSED_DIR, "adata_scored.h5ad")
    if not os.path.exists(in_path):
        print("ERROR:", in_path, "not found. Run 04_signature_scoring.py first.")
        sys.exit(1)

    print("\n[1/5] Loading scored object...")
    adata = sc.read_h5ad(in_path)
    n_clusters = adata.obs["leiden"].nunique()
    print(f"  Loaded: {adata.n_obs} cells, {n_clusters} clusters")
    print(f"  Involved: {(adata.obs['condition']=='Involved').sum()} | "
          f"Uninvolved: {(adata.obs['condition']=='Uninvolved').sum()}")
    if "norm_log" in adata.layers:
        adata.X = adata.layers["norm_log"]
    else:
        sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)

    # Primary DE: Involved vs Uninvolved (all cells)
    print("\n[2/5] Running Involved vs Uninvolved Wilcoxon DE (all cells)...")
    sc.tl.rank_genes_groups(adata, groupby="condition", groups=["Involved"],
                            reference="Uninvolved", method="wilcoxon",
                            use_raw=False, key_added="rank_genes_inv_vs_uni", pts=True)
    result = adata.uns["rank_genes_inv_vs_uni"]
    inv_vs_uni = pd.DataFrame({
        "gene":    result["names"]["Involved"],
        "score":   result["scores"]["Involved"],
        "pval_adj": result["pvals_adj"]["Involved"],
        "log2fc":  result["logfoldchanges"]["Involved"],
    }).sort_values("score", ascending=False)
    inv_vs_uni.to_csv(os.path.join(PROCESSED_DIR, "de_Involved_vs_Uninvolved.csv"), index=False)
    print(f"  Top upregulated: {inv_vs_uni.head(5)['gene'].tolist()}")
    print(f"  Top downregulated: {inv_vs_uni.tail(5)['gene'].tolist()}")

    # Cluster-vs-rest DE
    print("\n[3/5] Running cluster-vs-rest Wilcoxon DE...")
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon",
                            use_raw=False, key_added="rank_genes_groups", pts=True)
    result2 = adata.uns["rank_genes_groups"]
    rows = []
    for cl in result2["names"].dtype.names:
        genes  = result2["names"][cl]
        scores = result2["scores"][cl]
        pvals  = result2["pvals_adj"][cl]
        df_cl = pd.DataFrame({"cluster": cl, "gene": genes, "score": scores, "pval_adj": pvals})
        rows.extend([df_cl.nlargest(N_TOP_GENES, "score").assign(direction="up"),
                     df_cl.nsmallest(N_TOP_GENES, "score").assign(direction="down")])
    top_genes_df = pd.concat(rows, ignore_index=True)
    print(f"  Extracted {len(top_genes_df)} gene-cluster pairs")

    # Epithelial-specific Involved vs Uninvolved
    print("\n[4/5] Running Involved vs Uninvolved DE within epithelial clusters...")
    if "cell_type" in adata.obs.columns:
        epi_mask = adata.obs["cell_type"].str.contains(
            "colonocyte|goblet|stem|paneth|enteroendocrine", na=False)
        adata_epi = adata[epi_mask].copy()
        if adata_epi.n_obs > 100 and (adata_epi.obs["condition"] == "Involved").sum() > 50:
            sc.tl.rank_genes_groups(adata_epi, groupby="condition", groups=["Involved"],
                                    reference="Uninvolved", method="wilcoxon",
                                    use_raw=False, key_added="rank_epi_inv")
            epi_result = adata_epi.uns["rank_epi_inv"]
            epi_de = pd.DataFrame({
                "gene":    epi_result["names"]["Involved"],
                "score":   epi_result["scores"]["Involved"],
                "pval_adj": epi_result["pvals_adj"]["Involved"],
            }).sort_values("score", ascending=False)
            epi_de.to_csv(os.path.join(PROCESSED_DIR, "de_epithelial_Involved_vs_Uninvolved.csv"), index=False)
            print(f"  Epithelial DE top genes: {epi_de.head(5)['gene'].tolist()}")
        else:
            print("  Insufficient epithelial cells for focused DE.")

    # Pro-IBD cluster focused DE
    pro_clusters = list(adata.uns.get("pro_ibd_clusters", []))
    if pro_clusters:
        pro_str = [str(c) for c in pro_clusters]
        adata.obs["pro_ibd_group"] = adata.obs["leiden"].apply(
            lambda x: "pro_ibd" if str(x) in pro_str else "other")
        sc.tl.rank_genes_groups(adata, groupby="pro_ibd_group",
                                groups=["pro_ibd"], reference="other",
                                method="wilcoxon", use_raw=False,
                                key_added="rank_genes_proibd")
        pr = adata.uns["rank_genes_proibd"]
        pr_df = pd.DataFrame({
            "gene":    pr["names"]["pro_ibd"],
            "score":   pr["scores"]["pro_ibd"],
            "pval_adj": pr["pvals_adj"]["pro_ibd"],
        }).sort_values("score", ascending=False)
        pr_df.to_csv(os.path.join(PROCESSED_DIR, "de_proibd_vs_rest.csv"), index=False)
        print("  Pro-IBD cluster DE saved.")

    print("\n[5/5] Saving...")
    de_path = os.path.join(PROCESSED_DIR, "de_top_genes.csv")
    top_genes_df.to_csv(de_path, index=False)
    adata.write_h5ad(os.path.join(PROCESSED_DIR, "adata_de.h5ad"))
    print("\n" + "=" * 60)
    print("Script 05 complete.")
    print(f"  Involved vs Uninvolved DE: de_Involved_vs_Uninvolved.csv")
    print(f"  Cluster DE: {de_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
