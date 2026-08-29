# ============================================================
# TASK 1 — Final Regression Comparison Table
# Baseline + Tuned Models
# ============================================================

import pandas as pd
from pathlib import Path

script_dir = Path(__file__).resolve().parent

output_dir = script_dir / "task1_final_regression_comparison_outputs"
output_dir.mkdir(exist_ok=True)

comparison_data = [
    {
        "Model": "Random Forest",
        "Model_Type": "Baseline",
        "Train_R2": 0.9940,
        "Test_R2": 0.8879,
        "Train_RMSE_mL_CH4_gVS": 10.7558,
        "Test_RMSE_mL_CH4_gVS": 46.5112,
        "Train_MAE_mL_CH4_gVS": 5.2583,
        "Test_MAE_mL_CH4_gVS": 20.7243,
        "Comment": "Strong baseline; lowest baseline test MAE"
    },
    {
        "Model": "SVR",
        "Model_Type": "Baseline",
        "Train_R2": 0.9507,
        "Test_R2": 0.8707,
        "Train_RMSE_mL_CH4_gVS": 30.7132,
        "Test_RMSE_mL_CH4_gVS": 49.9422,
        "Train_MAE_mL_CH4_gVS": 18.8897,
        "Test_MAE_mL_CH4_gVS": 30.9455,
        "Comment": "Weakest baseline model"
    },
    {
        "Model": "ANN / MLPRegressor",
        "Model_Type": "Baseline",
        "Train_R2": 0.9910,
        "Test_R2": 0.9043,
        "Train_RMSE_mL_CH4_gVS": 13.1013,
        "Test_RMSE_mL_CH4_gVS": 42.9736,
        "Train_MAE_mL_CH4_gVS": 8.7067,
        "Test_MAE_mL_CH4_gVS": 21.4588,
        "Comment": "Best baseline test R2 and RMSE"
    },
    {
        "Model": "XGBoost",
        "Model_Type": "Baseline",
        "Train_R2": 0.9970,
        "Test_R2": 0.8979,
        "Train_RMSE_mL_CH4_gVS": 7.5383,
        "Test_RMSE_mL_CH4_gVS": 44.3719,
        "Train_MAE_mL_CH4_gVS": 5.3440,
        "Test_MAE_mL_CH4_gVS": 21.3583,
        "Comment": "Competitive baseline; some overfitting"
    },
    {
        "Model": "Tuned ANN / MLPRegressor",
        "Model_Type": "Tuned",
        "Train_R2": 0.9924,
        "Test_R2": 0.9092,
        "Train_RMSE_mL_CH4_gVS": 12.0493,
        "Test_RMSE_mL_CH4_gVS": 41.8471,
        "Train_MAE_mL_CH4_gVS": 7.9858,
        "Test_MAE_mL_CH4_gVS": 20.6665,
        "Comment": "Best final model"
    },
    {
        "Model": "Tuned XGBoost",
        "Model_Type": "Tuned",
        "Train_R2": 0.9983,
        "Test_R2": 0.8971,
        "Train_RMSE_mL_CH4_gVS": 5.6791,
        "Test_RMSE_mL_CH4_gVS": 44.5553,
        "Train_MAE_mL_CH4_gVS": 3.9861,
        "Test_MAE_mL_CH4_gVS": 21.5418,
        "Comment": "Training improved, but test performance did not improve"
    }
]

comparison_df = pd.DataFrame(comparison_data)

# Rank by Test R2 descending, then Test RMSE ascending
comparison_df = comparison_df.sort_values(
    by=["Test_R2", "Test_RMSE_mL_CH4_gVS"],
    ascending=[False, True]
).reset_index(drop=True)

comparison_df.insert(0, "Rank", range(1, len(comparison_df) + 1))

# Save files
comparison_df.to_csv(
    output_dir / "task1_final_regression_comparison.csv",
    index=False
)

comparison_df.to_excel(
    output_dir / "task1_final_regression_comparison.xlsx",
    index=False
)

print("\nTASK 1 final regression comparison table created successfully.")
print("\nFinal comparison table:")
print(comparison_df)

print("\nBest final model:")
print(comparison_df.iloc[0]["Model"])

print("\nFiles saved in:")
print(output_dir.resolve())