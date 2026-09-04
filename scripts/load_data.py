import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Metadatos normales
df_metadatos_normales = pd.read_csv(RAW / "GSE62944_06_01_15_TCGA_24_Normal_CancerType_Samples.txt",
                                    sep="\t", header=None,
                                    index_col=0, names=["SampleID", "CancerType"])
df_metadatos_normales_brca = df_metadatos_normales[df_metadatos_normales["CancerType"] == "BRCA"]

# Metadatos tumores
df_metadatos_tumores = pd.read_csv(RAW / "GSE62944_06_01_15_TCGA_24_CancerType_Samples.txt",
                                   sep="\t", header=None,
                                   index_col=0, names=["SampleID", "CancerType"])
df_metadatos_tumores_brca = df_metadatos_tumores[df_metadatos_tumores["CancerType"] == "BRCA"]

# Counts normales
df_counts_normales = pd.read_csv(RAW / "GSM1697009_06_01_15_TCGA_24.normal_Rsubread_FeatureCounts.txt",
                                 sep="\t", index_col=0)
df_counts_normales_brca = df_counts_normales[df_metadatos_normales_brca.index]

# Counts tumores
df_counts_tumores = pd.read_csv(RAW / "GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_FeatureCounts.txt",
                                sep="\t", index_col=0)
df_counts_tumores_brca = df_counts_tumores[df_metadatos_tumores_brca.index]

# Unir matrices
df_counts_unidas = pd.concat([df_counts_normales_brca, df_counts_tumores_brca], axis=1)

# Condición
df_metadatos_normales_brca = df_metadatos_normales_brca.copy()
df_metadatos_tumores_brca = df_metadatos_tumores_brca.copy()
df_metadatos_normales_brca["condition"] = "normal"
df_metadatos_tumores_brca["condition"] = "tumor"
df_metadatos_brca = pd.concat([df_metadatos_normales_brca, df_metadatos_tumores_brca])

# Guardar
df_counts_unidas.to_csv(PROCESSED / "counts_brca.csv")
df_metadatos_brca.to_csv(PROCESSED / "metadata_brca.csv")

print("Datos guardados correctamente.")
print(f"Counts shape: {df_counts_unidas.shape}")
print(f"Metadata shape: {df_metadatos_brca.shape}")