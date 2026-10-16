**AnaeroVision**

*AI-powered decision-support framework for anaerobic digestion.*

AnaeroVision uses machine learning to predict methane yield, detect inhibition risk, and classify probable inhibition mechanisms from anaerobic-digestion operating conditions.

---

**Features**

- Methane yield prediction
- Inhibition risk detection
- Inhibition type classification
- Confidence-based inhibition interpretation
- Streamlit graphical interface

---

**Model Pipeline**

| Task | Objective | Final Model |
|---|---|---|
| Task 1 | Methane yield prediction | Tuned ANN / MLPRegressor |
| Task 2 | Inhibition risk detection | ANN / MLPClassifier |
| Task 3 | Inhibition type classification | Tuned Random Forest |

```text
Process Conditions
       ↓
Methane Yield Prediction
       ↓
Inhibition Risk Detection
       ↓
If inhibition is detected
       ↓
Inhibition Type Classification
```

The inhibition-type model identifies:

- Ammonia inhibition
- LCFA inhibition
- VFA accumulation
- Salinity inhibition
- Other inhibition

---

**Inputs**

AnaeroVision uses the following main process variables:

- Substrate
- Reactor type
- Temperature
- pH
- Hydraulic retention time (HRT)
- Organic loading rate (OLR)
- Inoculum-to-substrate ratio (ISR)
- Total ammonia nitrogen (TAN)
- Volatile fatty acids (VFA)

---

**Dataset**

The operational dataset (`DATASET/Operational-dataset.xlsx`) contains 1500 records:

| Record type | `Source` value | Records |
|---|---|---:|
| Extracted from peer-reviewed anaerobic digestion studies | `Original` | 316 |
| Generated records added to compensate for the scarcity of published data (not experimental measurements, not attributed to any study) | `Generated` | 184 |
| Synthetic records produced by Gaussian Copula augmentation | `Synthetic_Gaussian` | 1000 |

Only the `Original` records are literature-derived. The complete list of source studies is given in the Supplementary Information of the associated manuscript.

---

**Model Performance**

| Task | Final model | Performance |
|---|---|---:|
| Methane yield prediction | Tuned ANN/MLP regressor | Test R² = 0.909; RMSE = 41.85 mL CH₄/gVS; MAE = 20.67 mL CH₄/gVS |
| Inhibition risk detection | ANN/MLP classifier (threshold 0.14) | Accuracy = 91.22%; F1 (inhibited class) = 0.848 |
| Inhibition type classification | Tuned Random Forest | Out-of-fold accuracy = 92.73%; macro F1 = 0.937 |

Performance values are those reported in the associated manuscript. Test sets include literature-derived, generated, and synthetic records.

---

**Project Structure**

```text
AnaeroVision/
├── DATASET/                                 # Operational dataset (1500 records)
├── Exploratory Data Analysis/               # EDA script and outputs
├── TASK1-Methane yield regression/          # Task 1 scripts, outputs, final model
├── TASK2-Binary inhibtion classification/   # Task 2 scripts, outputs, final model
├── TASK3-Inhibition type classification/    # Task 3 scripts, outputs, final model
├── Streamlit interface/anaerovision3.py     # AnaeroVision application
└── README.md
```

---

**Installation**

Clone the repository:

```powershell
git clone https://github.com/Louay22-stack/AnaeroVision.git
cd AnaeroVision
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install streamlit pandas numpy scikit-learn category_encoders joblib xgboost
```

---

**Run AnaeroVision**

The application loads the final models from folders placed next to `anaerovision3.py`. Copy the three final model folders into `Streamlit interface/` before running:

- `TASK1-Methane yield regression/final_task1_methane_yield_model/`
- `TASK2-Binary inhibtion classification/final_task2_inhibition_risk_model/`
- `TASK3-Inhibition type classification/final_task3_inhibition_type_model/`

Then run:

```powershell
cd "Streamlit interface"
streamlit run anaerovision3.py
```

Open the local Streamlit address displayed in the terminal.

---

**Research Context**

AnaeroVision was developed as part of research on **AI-assisted optimization of anaerobic digestion for biogas and methane production**.

The project combines:

**Anaerobic Digestion · Bioprocess Engineering · Environmental Biotechnology · Machine Learning**

---

**Author**

**Louay Ouhibi**

GitHub: [Louay22-stack](https://github.com/Louay22-stack)

---

**Disclaimer**

AnaeroVision is intended for research and decision-support purposes. Model predictions should be interpreted together with experimental measurements and professional process-engineering assessment.

---

**License**

See the [LICENSE](LICENSE) file for licensing information.