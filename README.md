# RNA-seq BRCA Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![pydeseq2](https://img.shields.io/badge/pydeseq2-DEG_Analysis-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data_Processing-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

End-to-end bioinformatics pipeline for differential gene expression analysis and binary classification of breast cancer samples (TCGA-BRCA), integrating DESeq2 statistical analysis with Machine Learning.

---

## Aim of this project

Build a complete bioinformatics pipeline covering raw count data loading, preprocessing, differential expression analysis (DEA), and machine learning classification, from raw RNA-seq counts to biological interpretation.

Models evaluated: Logistic Regression (baseline), Random Forest (fine-tuned).

---

## Project Structure

```
rnaseq-brca-pipeline/
│
├── data/
│   ├── raw/                        # Raw GEO files (not tracked by Git)
│   └── processed/                  # Processed matrices (not tracked by Git)
│
├── results/
│   └── figures/                    # Plots and visualizations (.png)
│
├── scripts/
│   ├── load_data.py                # Data loading and merging
│   ├── preprocessing.py            # Filtering, normalization, VST, PCA, DEG
│   └── ml_pipeline.py              # ML classification and feature importance
│
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/moralesgomez-dev/rnaseq-brca-pipeline.git
cd rnaseq-brca-pipeline
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
.\.venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the raw data

Download the following files from [NCBI GEO — GSE62944](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE62944) and place them in `data/raw/`:

- `GSE62944_06_01_15_TCGA_24_Normal_CancerType_Samples.txt.gz`
- `GSE62944_06_01_15_TCGA_24_CancerType_Samples.txt.gz`
- `GSM1697009_06_01_15_TCGA_24.normal_Rsubread_FeatureCounts.txt.gz`
- `GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_FeatureCounts.txt.gz`

### 5. Run the pipeline

```bash
python scripts/load_data.py
python scripts/preprocessing.py
python scripts/ml_pipeline.py
```

---

## Pipeline Overview

### Load Data (`load_data.py`)
- Loads raw count matrices and clinical metadata from GEO
- Filters BRCA samples from a multi-cancer dataset (24 cancer types)
- Merges tumor and normal count matrices
- Assigns condition labels (`tumor` / `normal`)
- Saves processed files to `data/processed/`

### Preprocessing & DEG Analysis (`preprocessing.py`)
- Low-count gene filtering (sum < 10 across all samples)
- DESeq2 normalization via `pydeseq2` (`DeseqDataSet`)
- Variance Stabilizing Transformation (VST) for visualization and ML
- PCA to assess sample separation by condition
- Differential expression analysis (`DeseqStats`) with Wald test
- DEG filtering: padj < 0.05 and |log2FC| > 1
- Volcano plot and heatmap generation

### ML Pipeline (`ml_pipeline.py`)
- Binary classification: tumor vs normal using VST expression values
- `StratifiedGroupKFold` cross-validation to prevent patient-level data leakage
- Models: Logistic Regression (baseline) and Random Forest
- Evaluation: tumor recall, normal recall, ROC AUC, balanced accuracy
- Fine-tuning with `RandomizedSearchCV`
- Permutation test (sanity check) to validate model learning
- Feature importance analysis and biological interpretation

---

## Dataset

| Parameter | Value |
|-----------|-------|
| Source | TCGA-BRCA via NCBI GEO (GSE62944) |
| Tumor samples | 1,119 |
| Normal samples | 113 |
| Genes (after filtering) | 22,747 |
| Class ratio | ~10:1 (tumor:normal) |

---

## Results

### Differential Expression Analysis

The DEG analysis covered more than 20,000 genes across 1,232 samples. After filtering (padj < 0.05 and |log2FC| > 1), **5,791 differentially expressed genes** were identified.

**Top upregulated genes** (tumor vs normal):

| Gene | log2FC | Biological role |
|------|--------|-----------------|
| COL10A1 | +7.10 | Extracellular matrix remodeling, invasion |
| MMP11 | +6.26 | Tumor-immune signaling, metastasis |
| COL11A1 | +6.16 | Extracellular matrix remodeling, invasion |
| MMP13 | +6.16 | Extracellular matrix remodeling, invasion |
| NEK2 | +4.29 | Cell cycle dysregulation |

**Top downregulated genes** (tumor vs normal):

| Gene | log2FC | Biological role |
|------|--------|-----------------|
| LPL | -4.74 | Lipid metabolism, M1 macrophage polarization |
| GPAM | -4.43 | Lipid metabolism, tumor suppressor |
| PPARG | -3.40 | Lipid metabolism, tumor suppressor |

All five upregulated genes promote tumor progression through different mechanisms and are consistent with current literature. The three downregulated genes are tumor suppressors involved in lipid metabolism whose dysregulation creates an optimal environment for tumor progression.

### Model Performance

| Model | ROC AUC (CV) | Balanced Accuracy (CV) | Tumor Recall (CV) | Normal Recall (CV) |
|-------|-------------|----------------------|-------------------|--------------------|
| Logistic Regression | 0.9993 ± 0.0011 | 0.9817 ± 0.0224 | 0.9966 ± 0.0027 | 0.9667 ± 0.0444 |
| Random Forest | 0.9998 ± 0.0004 | 0.9606 ± 0.0282 | 0.9978 ± 0.0027 | 0.9234 ± 0.0560 |

**Tuned Random Forest — test set results:**

| Metric | Value |
|--------|-------|
| ROC AUC | 0.9992 |
| Accuracy | 0.99 |
| Normal precision/recall | 0.95 / 0.95 |
| Tumor precision/recall | 1.00 / 1.00 |

Permutation test (shuffled labels) produced a mean AUC of ~0.49, confirming that the model learns real biological signal and not noise.

### Feature Importance

The top genes by Random Forest importance include **SDPR** and **PAMR1**, both significantly downregulated in breast cancer and acting as tumor suppressors by inhibiting proliferation, migration, and invasion.

Notably, these genes do not appear among the most statistically significant DEGs — which illustrates an important distinction: DESeq2 identifies genes with consistent group-level change, while Random Forest identifies genes with high individual discriminative power. Both criteria are complementary and not necessarily coincident.

---

## Limitations

- **Class imbalance**: ~10:1 ratio (1,119 tumors vs 113 normals)
- **Single cohort**: no external validation was performed
- **PCA overlap**: 80 tumor samples overlapped with the normal cluster in PCA, suggesting real biological heterogeneity — these were retained in the analysis as representative of clinical variability
- **Task simplicity**: tumor vs normal classification based on RNA-seq expression is inherently straightforward given the magnitude of expression differences between tissues. More clinically relevant tasks would include molecular subtype classification or treatment response prediction

---

## Note on Model Performance

The extraordinary model performance (ROC AUC ~0.9992) should not be interpreted as a clinically relevant result. Classifying tumor vs adjacent normal tissue is an intrinsically simple task, as the gene expression differences between both tissues are massive and well established.

The real value of this project lies in building the process: a complete pipeline from raw data to biological interpretation, integrating classical statistical analysis (DESeq2) with machine learning.

---

## References

Aboudi, N. E., Ouardi, F., Ababou, M., Abdelilah, L., Elbiad, O., & Badaoui, B. (2025). Differential gene expression and gene ontology associated with breast cancer development and progression: A meta-analysis study. *Eurasian Journal of Medicine and Oncology*. https://doi.org/10.36922/ejmo025060025

Ali, R., Sultan, A., Ishrat, R., Haque, S., Khan, N., & Prieto, M. (2023). Identification of New Key Genes and Their Association with Breast Cancer Occurrence and Poor Survival Using In Silico and In Vitro Methods. *Biomedicines, 11*. https://doi.org/10.3390/biomedicines11051271

Anuraga, G., Wang, W., Phan, N., Ton, N. T. A., Ta, H. D. K., Prayugo, F. B., Xuan, D. T. M., Ku, S.-C., Wu, Y., Andriani, V., Athoillah, M., Lee, K.-H., & Wang, C.-Y. (2021). Potential Prognostic Biomarkers of NIMA (Never in Mitosis, Gene A)-Related Kinase (NEK) Family Members in Breast Cancer. *Journal of Personalized Medicine, 11*. https://doi.org/10.3390/jpm11111089

Brockmöller, S. F., Bucher, E., Müller, B., Budczies, J., Hilvo, M., Griffin, J., Orešič, M., Kallioniemi, O., Iljin, K., Loibl, S., Darb-Esfahani, S., Sinn, B., Klauschen, F., Prinzler, J., Bangemann, N., Ismaeel, F., Fiehn, O., Dietel, M., & Denkert, C. (2012). Integration of metabolomics and expression of glycerol-3-phosphate acyltransferase (GPAM) in breast cancer — link to patient survival, hormone receptor status, and metabolic profiling. *Journal of Proteome Research, 11*(2), 850–860. https://doi.org/10.1021/pr200685r

Cheng, T., Chen, P., Chen, J.-Y., Deng, Y.-N., & Huang, C. (2022). Landscape Analysis of Matrix Metalloproteinases Unveils Key Prognostic Markers for Patients With Breast Cancer. *Frontiers in Genetics, 12*. https://doi.org/10.3389/fgene.2021.809600

Li, D.-H., Liu, X.-K., Tian, X., Liu, F., Yao, X., & Dong, J. (2023). PPARG: A promising therapeutic target in breast cancer and regulation by natural drugs. *PPAR Research, 2023*. https://doi.org/10.1155/2023/4481354

Liu, Z., Gao, Z., Li, B., Li, J., Ou, Y.-F., Yu, X., Zhang, Z.-W., Liu, S., Fu, X., Jin, H., Wu, J., Sun, S., & Wu, Q. (2022). Lipid-associated macrophages in the tumor-adipose microenvironment facilitate breast cancer progression. *Oncoimmunology, 11*. https://doi.org/10.1080/2162402x.2022.2085432

Tian, Y., Yu, Y., Hou, L., Chi, J., Mao, J., Xia, L., Wang, X., Wang, P., & Cao, X. (2016). Serum deprivation response inhibits breast cancer progression by blocking transforming growth factor‐β signaling. *Cancer Science, 107*, 274–280. https://doi.org/10.1111/cas.12879

Vargas, A., Reed, A., Waddell, N., Lane, A., Reid, L., Smart, C., Cocciardi, S., Silva, L., Song, S., Chenevix-Trench, G., Simpson, P., & Lakhani, S. (2012). Gene expression profiling of tumour epithelial and stromal compartments during breast cancer progression. *Breast Cancer Research and Treatment, 135*, 153–165. https://doi.org/10.1007/s10549-012-2123-4

Wang, Q., & Shao, J. (2026). Screening of the key gene PAMR1 in breast cancer and its immunological and pharmacogenomic characteristics. *Discover Oncology*. https://doi.org/10.1007/s12672-026-05312-6

Yang, L., Fang, X., Liu, X., Liu, Y., & Zhao, S. (2025). The lipid metabolism-associated immune gene LPL promotes M1 macrophage polarization and inhibits breast cancer progression. *Tissue and Cell, 97*, 103071. https://doi.org/10.1016/j.tice.2025.103071

Yao, K., Zhu, X., Zhao, T., Liao, S., Ji, L., Wei, Z., Li, Y., Tian, J., Ding, X., Jun, Z., Qing, B., & Jun, L. (2023). Multidimensional analysis to elucidate the possible mechanism of bone metastasis in breast cancer. *BMC Cancer, 23*. https://doi.org/10.1186/s12885-023-11588-6

Zhou, W., Li, Y., Gu, D., Xu, J., Wang, R., Wang, H., & Liu, C. (2022). High expression COL10A1 promotes breast cancer progression and predicts poor prognosis. *Heliyon, 8*. https://doi.org/10.1016/j.heliyon.2022.e11083

Zhuang, Y., Li, X., Zhan, P., Pi, G., & Wen, G. (2021). MMP11 promotes the proliferation and progression of breast cancer through stabilizing Smad2 protein. *Oncology Reports, 45*. https://doi.org/10.3892/or.2021.7967

---

## Author

**AlejandroMoralesGomezDev**
- GitHub: [moralesgomez-dev](https://github.com/moralesgomez-dev)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
