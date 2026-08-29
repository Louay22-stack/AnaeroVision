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

**Model Performance**

| Task | Performance |
|---|---:|
| Methane yield prediction | Test R² = 0.87 |
| Inhibition risk detection | Accuracy = 97.67% |
| Inhibition type classification | OOF Accuracy = 92.73% |
| Inhibition type classification | Macro F1 = 93.68% |

---

**Project Structure**

```text
AnaeroVision/
├── anaerovision3.py
├── final_task1_methane_yield_model/
├── final_task2_inhibition_risk_model/
├── final_task3_inhibition_type_model/
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
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
pip install -r requirements.txt
```

---

**Run AnaeroVision**

```powershell
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