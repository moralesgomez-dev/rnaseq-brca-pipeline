import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pydeseq2.dds import DefaultInference, DeseqDataSet
from pydeseq2.ds import DeseqStats
from sklearn.decomposition import PCA

ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "results" / "figures"

# Cargar datos
counts_brca = pd.read_csv(PROCESSED / "counts_brca.csv", index_col=0).T
metadata_brca = pd.read_csv(PROCESSED / "metadata_brca.csv", index_col=0)

# Filtrar
samples_to_keep = ~metadata_brca.condition.isna()
counts_brca_filtered = counts_brca.loc[samples_to_keep]
metadata_brca_filtered = metadata_brca.loc[samples_to_keep]

genes_to_keep = counts_brca_filtered.columns[counts_brca_filtered.sum(axis=0) >= 10]
counts_brca_filtered = counts_brca_filtered[genes_to_keep]

# DESeqDataSet
inference = DefaultInference(n_cpus=8)
dds = DeseqDataSet(counts=counts_brca_filtered, metadata=metadata_brca_filtered,
                   design="~condition", refit_cooks=True, inference=inference)
dds.deseq2()

# VST
dds.vst()
vst_data = dds.layers["vst_counts"]
vst_data_df = pd.DataFrame(vst_data, index=dds.obs_names, columns=dds.var_names)

# PCA
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(vst_data_df)
pca_df = pd.DataFrame(pca_coords, columns=["PC1", "PC2"], index=vst_data_df.index)
pca_df["condition"] = metadata_brca_filtered["condition"]

fig, ax = plt.subplots(figsize=(8, 6))
for condition, color in [("tumor", "salmon"), ("normal", "steelblue")]:
    mask = pca_df["condition"] == condition
    ax.scatter(pca_df.loc[mask, "PC1"], pca_df.loc[mask, "PC2"],
               label=condition, alpha=0.6, s=20, color=color)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("PCA — BRCA tumor vs normal")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES / "pca_brca.png", dpi=150)
plt.show()

# DEG
stat_res = DeseqStats(dds, contrast=["condition", "tumor", "normal"])
stat_res.summary()
df_deg = stat_res.results_df
df_deg_filtered = df_deg[(df_deg['padj'] < 0.05) & (df_deg['log2FoldChange'].abs() > 1)]

# Visualizaciones
def plot_volcano(res_df):
    df = res_df.copy().dropna(subset=["padj", "log2FoldChange"])
    df["negLog10Padj"] = -np.log10(df["padj"] + 1e-300)
    df["padj_cat"] = pd.cut(df["padj"], bins=[-np.inf, 0.01, 0.05, 0.2, np.inf],
                            labels=["< 0.01", "0.01–0.05", "0.05–0.2", "≥ 0.2"])
    colors = {"< 0.01": "#1B9E77", "0.01–0.05": "#dbf229", "0.05–0.2": "#fcb103", "≥ 0.2": "#fc0303"}
    fig, ax = plt.subplots(figsize=(10, 7))
    for cat, color in colors.items():
        mask = df["padj_cat"] == cat
        ax.scatter(df.loc[mask, "log2FoldChange"], df.loc[mask, "negLog10Padj"],
                   c=color, alpha=0.7, s=10, label=cat)
    ax.axvline(x=-1, linestyle="--", color="grey", linewidth=0.8)
    ax.axvline(x=1, linestyle="--", color="grey", linewidth=0.8)
    ax.axhline(y=-np.log10(0.05), linestyle="--", color="grey", linewidth=0.8)
    ax.set_xlabel("log2 Fold Change")
    ax.set_ylabel("-log10(padj)")
    ax.set_title("Volcano Plot — BRCA tumor vs normal")
    ax.legend(title="padj")
    plt.tight_layout()
    plt.savefig(FIGURES / "volcano_brca.png", dpi=150)
    plt.show()

def plot_top_heatmap_mean(res_df, vst_df, metadata_df, top_n=50):
    res_sig = res_df.dropna(subset=["padj"]).sort_values("padj")
    top_genes = res_sig.head(top_n).index.tolist()
    mat = vst_df[top_genes].T
    tumor_samples = metadata_df[metadata_df["condition"] == "tumor"].index
    normal_samples = metadata_df[metadata_df["condition"] == "normal"].index
    mat_mean = pd.DataFrame({
        "tumor": mat[tumor_samples].mean(axis=1),
        "normal": mat[normal_samples].mean(axis=1)
    })
    g = sns.clustermap(mat_mean, cmap="RdBu_r", center=0, z_score=0,
                       figsize=(6, 12), yticklabels=True)
    g.fig.suptitle(f"Top {top_n} DEGs — media por condición", y=1.02)
    g.savefig(FIGURES / "heatmap_brca.png", dpi=150)
    plt.show()

plot_volcano(df_deg_filtered)
plot_top_heatmap_mean(df_deg_filtered, vst_data_df, metadata_brca_filtered)