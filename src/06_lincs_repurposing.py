import os, sys, time, re
import numpy as np
import pandas as pd
import gseapy as gp
import scanpy as sc

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

N_TOP_GENES  = 150
TOP_PER_QUERY = 15
ENRICHR_DELAY = 1.0
ENRICHR_LIBRARIES = [
    "LINCS_L1000_Chem_Pert_up",
    "LINCS_L1000_Chem_Pert_down",
    "GO_Biological_Process_2023",
    "Reactome_2022",
    "KEGG_2021_Human",
]

def clean_compound_name(term):
    m = re.match(r'^LJP\d+\s+\S+\s+\S+?-(.+)-[\d.]+$', term.strip())
    if m: return m.group(1).strip()
    return term.split("_")[0].strip() if term else term.strip()

def run_enrichr(query_id, up_genes, down_genes):
    results = []
    for direction, genes, reversal_lib in [
        ("up",   up_genes,   "LINCS_L1000_Chem_Pert_down"),
        ("down", down_genes, "LINCS_L1000_Chem_Pert_up"),
    ]:
        if not genes: continue
        for lib in ENRICHR_LIBRARIES:
            try:
                enr = gp.enrichr(gene_list=genes, gene_sets=lib, outdir=None, verbose=False)
                df = enr.results.copy()
                if df.empty: continue
                df["query_id"]       = query_id
                df["query_direction"] = direction
                df["library"]        = lib
                if lib in ("LINCS_L1000_Chem_Pert_up", "LINCS_L1000_Chem_Pert_down"):
                    adj_p = df["Adjusted P-value"].clip(lower=1e-300)
                    sign  = 1.0 if lib == reversal_lib else -1.0
                    df["reversal_score"] = sign * (-np.log10(adj_p))
                    df["compound"] = df["Term"].apply(clean_compound_name)
                else:
                    df["reversal_score"] = 0.0
                    df["compound"] = df["Term"]
                results.append(df)
                time.sleep(ENRICHR_DELAY)
            except Exception as e:
                print(f"    WARNING: Enrichr failed for {query_id} {lib}: {e}")
                time.sleep(ENRICHR_DELAY * 2)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def deduplicate_and_rank(raw_df):
    if raw_df.empty: return pd.DataFrame()
    lincs_mask = raw_df["library"].str.startswith("LINCS_L1000")
    lincs_df   = raw_df[lincs_mask & (raw_df["reversal_score"] > 0)].copy()
    if lincs_df.empty:
        print("  WARNING: No positive LINCS reversal scores.")
        return pd.DataFrame()
    top = lincs_df.sort_values("reversal_score", ascending=False).groupby("query_id").head(TOP_PER_QUERY)
    agg = top.groupby("compound").agg(
        max_reversal_score=("reversal_score", "max"),
        n_queries=("query_id", "nunique"),
        queries=("query_id", lambda x: ",".join(sorted(set(x.astype(str))))),
        best_query=("query_id", lambda x: x.loc[x.index[top.loc[x.index, "reversal_score"].argmax()]]),
    ).reset_index()
    return agg.sort_values("max_reversal_score", ascending=False)

def main():
    print("=" * 60)
    print("CROHN'S DISEASE scRNA-seq PIPELINE")
    print("Script 06: LINCS L1000 Reversal Scoring")
    print("Primary query: Involved vs Uninvolved DE")
    print("=" * 60)

    inv_path = os.path.join(PROCESSED_DIR, "de_Involved_vs_Uninvolved.csv")
    de_path  = os.path.join(PROCESSED_DIR, "de_top_genes.csv")
    if not os.path.exists(inv_path):
        print("ERROR:", inv_path, "not found. Run 05_differential_expression.py first.")
        sys.exit(1)

    print("\n[1/4] Loading DE gene lists...")
    inv_vs_uni  = pd.read_csv(inv_path)
    top_genes_df = pd.read_csv(de_path)
    n_clusters   = top_genes_df["cluster"].nunique()
    print(f"  Involved vs Uninvolved DE: {len(inv_vs_uni)} genes")
    print(f"  Cluster DE: {len(top_genes_df)} gene-cluster pairs across {n_clusters} clusters")

    print("\n[2/4] Submitting to Enrichr...")
    all_results = []

    # Primary: Involved vs Uninvolved whole-tissue
    print("  Primary query: Involved vs Uninvolved reversal...")
    inv_up   = inv_vs_uni[inv_vs_uni["score"] > 0].head(N_TOP_GENES)["gene"].tolist()
    inv_down = inv_vs_uni[inv_vs_uni["score"] < 0].tail(N_TOP_GENES)["gene"].tolist()
    prim_res = run_enrichr("Inv_vs_Uni", inv_up, inv_down)
    if not prim_res.empty:
        all_results.append(prim_res)
        print(f"   {(prim_res['reversal_score'] > 0).sum()} reversal hits")

    # Epithelial-specific Involved vs Uninvolved
    epi_path = os.path.join(PROCESSED_DIR, "de_epithelial_Involved_vs_Uninvolved.csv")
    if os.path.exists(epi_path):
        epi_de = pd.read_csv(epi_path)
        epi_up   = epi_de[epi_de["score"] > 0].head(N_TOP_GENES)["gene"].tolist()
        epi_down = epi_de[epi_de["score"] < 0].tail(N_TOP_GENES)["gene"].tolist()
        print("  Epithelial Involved vs Uninvolved query...")
        epi_res = run_enrichr("Epithelial_Inv_vs_Uni", epi_up, epi_down)
        if not epi_res.empty:
            all_results.append(epi_res)
            print(f"   {(epi_res['reversal_score'] > 0).sum()} reversal hits")

    # Cluster-vs-rest queries
    for i, cl in enumerate(top_genes_df["cluster"].unique()):
        print(f"  Cluster {cl} ({i+1}/{n_clusters})...", end=" ", flush=True)
        up   = top_genes_df.loc[(top_genes_df["cluster"] == cl) & (top_genes_df["direction"] == "up"), "gene"].tolist()
        down = top_genes_df.loc[(top_genes_df["cluster"] == cl) & (top_genes_df["direction"] == "down"), "gene"].tolist()
        res  = run_enrichr(f"cluster_{cl}", up, down)
        if not res.empty:
            all_results.append(res)
            print(f"{(res['reversal_score'] > 0).sum()} reversal hits")
        else:
            print("no results")

    if not all_results:
        print("ERROR: No Enrichr results."); sys.exit(1)
    raw_results = pd.concat(all_results, ignore_index=True)
    raw_results.to_csv(os.path.join(PROCESSED_DIR, "lincs_results_raw.csv"), index=False)
    print(f"  Raw results: {len(raw_results)} rows")

    print("\n[3/4] Deduplicating and ranking...")
    candidates = deduplicate_and_rank(raw_results)
    if not candidates.empty:
        print(f"  {len(candidates)} unique compounds identified")
        print(candidates[["compound","max_reversal_score","n_queries"]].head(10).round(2).to_string(index=False))

    print("\n[4/4] Saving...")
    cand_path = os.path.join(PROCESSED_DIR, "lincs_candidates.csv")
    candidates.to_csv(cand_path, index=False)
    print("\n" + "=" * 60)
    print(f"Script 06 complete. -> {cand_path} ({len(candidates)} compounds)")
    print("=" * 60)

if __name__ == "__main__":
    main()
