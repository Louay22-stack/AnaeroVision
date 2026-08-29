# ============================================================
# EDA BEFORE SPLITTING — Methane Yield Regression
# 1) Pearson correlation heatmap
# 2) Flag features with |correlation| > 0.5
# 3) IQR outlier flagging/report
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# 0. Load dataset
# ------------------------------------------------------------

# Folder where this EDA.py script is located
script_dir = Path(__file__).resolve().parent

# Dataset path: the Excel file should be inside the same folder as EDA.py
file_path = script_dir / "Operational-dataset.xlsx"

# Backup in case your file is named Operational-dataset(1).xlsx
if not file_path.exists():
    file_path = script_dir / "Operational-dataset(1).xlsx"

if not file_path.exists():
    raise FileNotFoundError(
        "Dataset not found. Put 'Operational-dataset.xlsx' or "
        "'Operational-dataset(1).xlsx' in the same folder as EDA.py."
    )

df = pd.read_excel(file_path)

print("Dataset loaded from:")
print(file_path)
print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# Create output folder inside Desktop/X
output_dir = script_dir / "eda_outputs_before_splitting"
output_dir.mkdir(exist_ok=True)

print("\nEDA outputs will be saved in:")
print(output_dir.resolve())

# ------------------------------------------------------------
# 0.1 Remove columns that should NOT be used in EDA/model input
# ------------------------------------------------------------

excluded_cols = [
    "TAN_was_missing",
    "VFA_was_missing"
]

df_eda = df.drop(columns=[col for col in excluded_cols if col in df.columns]).copy()

print("\nExcluded from EDA and future model input:")
print([col for col in excluded_cols if col in df.columns])

# ------------------------------------------------------------
# 1. Define target and numeric features for regression EDA
# ------------------------------------------------------------

target_col = "Methane_Yield_mL_CH4_gVS"

numeric_features = [
    "Temperature_C",
    "pH",
    "HRT_days",
    "OLR_gVS_L_day",
    "ISR",
    "TAN_mg_L",
    "VFA_mg_L"
]

# Keep only columns that actually exist in the uploaded dataset
eda_numeric_cols = [col for col in numeric_features + [target_col] if col in df_eda.columns]

eda_numeric_df = df_eda[eda_numeric_cols].copy()

print("\nNumeric columns used for correlation EDA:")
print(eda_numeric_cols)

# ------------------------------------------------------------
# 2. Pearson correlation heatmap
# ------------------------------------------------------------

corr_matrix = eda_numeric_df.corr(method="pearson")

# Save correlation matrix as CSV
corr_matrix.to_csv(output_dir / "pearson_correlation_matrix.csv")

plt.figure(figsize=(12, 9))
plt.imshow(corr_matrix, aspect="auto")
plt.colorbar(label="Pearson correlation coefficient")

plt.xticks(
    ticks=np.arange(len(corr_matrix.columns)),
    labels=corr_matrix.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    ticks=np.arange(len(corr_matrix.index)),
    labels=corr_matrix.index
)

plt.title("Pearson Correlation Heatmap — Full Dataset Before Splitting")

# Add correlation values inside cells
for i in range(len(corr_matrix.index)):
    for j in range(len(corr_matrix.columns)):
        value = corr_matrix.iloc[i, j]
        if not np.isnan(value):
            plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / "pearson_correlation_heatmap.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 3. Correlation of each numeric feature with methane yield
# ------------------------------------------------------------

target_correlations = corr_matrix[target_col].drop(target_col).sort_values(
    key=lambda x: x.abs(),
    ascending=False
)

target_correlations_df = target_correlations.reset_index()
target_correlations_df.columns = [
    "Feature",
    "Pearson_correlation_with_methane_yield"
]

# Flag primary predictors with |correlation| > 0.5
target_correlations_df["Primary_predictor_abs_corr_gt_0_5"] = (
    target_correlations_df["Pearson_correlation_with_methane_yield"].abs() > 0.5
)

target_correlations_df.to_csv(
    output_dir / "methane_yield_feature_correlations.csv",
    index=False
)

print("\nCorrelation with Methane_Yield_mL_CH4_gVS:")
print(target_correlations_df)

primary_predictors = target_correlations_df[
    target_correlations_df["Primary_predictor_abs_corr_gt_0_5"]
]

print("\nPrimary predictors with |correlation| > 0.5:")
if primary_predictors.empty:
    print("No numeric feature has |correlation| > 0.5 with methane yield.")
else:
    print(primary_predictors)

# ------------------------------------------------------------
# 4. IQR-based outlier flagging
# ------------------------------------------------------------

outlier_cols = [
    "TAN_mg_L",
    "VFA_mg_L",
    "OLR_gVS_L_day",
    "Methane_Yield_mL_CH4_gVS"
]

outlier_cols = [col for col in outlier_cols if col in df_eda.columns]

outlier_report = []

# Important: use df_eda, not df, so TAN_was_missing and VFA_was_missing
# are not saved in the flagged output file.
df_outlier_flags = df_eda.copy()

for col in outlier_cols:
    series = df_eda[col].dropna()

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    flag_col = f"{col}_IQR_outlier"

    df_outlier_flags[flag_col] = (
        (df_eda[col] < lower_bound) | (df_eda[col] > upper_bound)
    )

    n_outliers = df_outlier_flags[flag_col].sum()
    n_non_missing = df_eda[col].notna().sum()

    outlier_percentage = (
        (n_outliers / n_non_missing) * 100
        if n_non_missing > 0
        else np.nan
    )

    outlier_report.append({
        "Feature": col,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower_bound": lower_bound,
        "Upper_bound": upper_bound,
        "Non_missing_values": n_non_missing,
        "Outlier_count": int(n_outliers),
        "Outlier_percentage": outlier_percentage
    })

outlier_report_df = pd.DataFrame(outlier_report)

outlier_report_df.to_csv(
    output_dir / "iqr_outlier_report.csv",
    index=False
)

df_outlier_flags.to_excel(
    output_dir / "dataset_with_iqr_outlier_flags.xlsx",
    index=False
)

print("\nIQR outlier report:")
print(outlier_report_df)

# ------------------------------------------------------------
# 5. Boxplots for outlier visualization (publication quality)
# ------------------------------------------------------------

# High-quality global rendering settings
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.15
plt.rcParams["svg.fonttype"] = "none"   # keep text editable in SVG
plt.rcParams["pdf.fonttype"] = 42       # embed proper (non-rasterized) fonts in PDF

# Separate sub-folder to keep the boxplots organized
boxplot_dir = output_dir / "boxplots_high_quality"
boxplot_dir.mkdir(exist_ok=True)

for col in outlier_cols:
    data = df_eda[col].dropna()

    # Square canvas; constrained_layout keeps title/labels/ticks inside
    # the figure bounds instead of overflowing past the edges.
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=300, constrained_layout=True)

    bp = ax.boxplot(
        data,
        vert=True,
        widths=0.35,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(facecolor="#4C72B0", edgecolor="black", linewidth=1.4, alpha=0.85),
        whiskerprops=dict(color="black", linewidth=1.4),
        capprops=dict(color="black", linewidth=1.4),
        flierprops=dict(
            marker="o",
            markerfacecolor="#C44E52",
            markeredgecolor="black",
            markersize=6,
            alpha=0.8,
        ),
    )

    ax.set_title(col, pad=10, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xticks([])
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)

    # Clear, evenly spaced y-axis numbers
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax.tick_params(axis="y", labelsize=10, length=0)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, pos: f"{val:,.1f}" if abs(val) < 1000 else f"{val:,.0f}")
    )

    # Save in multiple high-quality formats:
    # - PNG at 600 dpi (raster, universally compatible)
    # - PDF (vector, lossless, ideal for print/publication)
    # - SVG (vector, lossless, editable text/shapes)
    fig.savefig(boxplot_dir / f"boxplot_{col}.png", dpi=600)
    fig.savefig(boxplot_dir / f"boxplot_{col}.pdf")
    fig.savefig(boxplot_dir / f"boxplot_{col}.svg")

    plt.show()
    plt.close(fig)

# ------------------------------------------------------------
# 6. Final message
# ------------------------------------------------------------

print("\nEDA completed successfully.")
print(f"All outputs saved in folder: {output_dir.resolve()}")
print(f"High-quality boxplots (PNG 600dpi, PDF, SVG) saved in: {boxplot_dir.resolve()}")