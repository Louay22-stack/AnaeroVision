# <div align="center">

# 

# \# AnaeroVision

# 

# \### AI-Powered Anaerobic Digestion Process Intelligence

# 

# \*\*Machine-learning decision-support framework for methane-yield prediction, inhibition-risk detection, and inhibition-mechanism classification in anaerobic digestion systems.\*\*

# 

# \[!\[Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python\&logoColor=white)](https://www.python.org/)

# \[!\[Streamlit](https://img.shields.io/badge/Streamlit-Application-FF4B4B?logo=streamlit\&logoColor=white)](https://streamlit.io/)

# \[!\[Machine Learning](https://img.shields.io/badge/Machine%20Learning-AD%20Process%20Intelligence-2E7D7B)](#)

# \[!\[GitHub last commit](https://img.shields.io/github/last-commit/Louay22-stack/AnaeroVision)](https://github.com/Louay22-stack/AnaeroVision)

# \[!\[GitHub license](https://img.shields.io/github/license/Louay22-stack/AnaeroVision)](https://github.com/Louay22-stack/AnaeroVision)

# 

# </div>

# 

# \---

# 

# \## Overview

# 

# \*\*AnaeroVision\*\* is an AI-driven decision-support framework developed for the analysis of \*\*anaerobic digestion (AD)\*\* systems.

# 

# The platform integrates three complementary machine-learning tasks into a single cascaded workflow:

# 

# 1\. \*\*Methane Yield Prediction\*\*

# 2\. \*\*Inhibition Risk Detection\*\*

# 3\. \*\*Inhibition Type Classification\*\*

# 

# The objective is to transform anaerobic-digestion operating conditions into interpretable predictions that can support process-performance assessment, early instability screening, and inhibition diagnosis.

# 

# AnaeroVision combines \*\*bioprocess engineering, environmental biotechnology, and machine learning\*\* within a user-oriented Streamlit interface.

# 

# > \*\*AnaeroVision is a research-oriented decision-support tool. It is not a substitute for laboratory analysis, process instrumentation, or professional engineering assessment.\*\*

# 

# \---

# 

# \## Prediction Workflow

# 

# AnaeroVision follows a cascaded machine-learning architecture.

# 

# ```mermaid

# flowchart LR

# &#x20;   A\[AD Process Conditions] --> B\[Data Preprocessing]

# 

# &#x20;   B --> C\[Task 1<br/>Methane Yield Prediction]

# &#x20;   B --> D\[Task 2<br/>Inhibition Risk Detection]

# 

# &#x20;   C --> E\[Predicted CH4 Yield]

# 

# &#x20;   D --> F{Inhibition Risk?}

# 

# &#x20;   F -->|No| G\[No inhibition risk detected]

# &#x20;   F -->|Yes| H\[Task 3<br/>Inhibition Type Classification]

# 

# &#x20;   H --> I\[Predicted Mechanism]

# &#x20;   H --> J\[Confidence Score]

# &#x20;   H --> K\[Class Probabilities]

# ```

# 

# The third model is activated \*\*only when the inhibition-risk model detects a potential inhibition event\*\*.

# 

# This architecture allows AnaeroVision to provide three complementary levels of process information:

# 

# ```text

# Expected methane production

# &#x20;           ↓

# Potential inhibition risk

# &#x20;           ↓

# Probable inhibition mechanism

# ```

# 

# \---

# 

# \## Machine-Learning Tasks

# 

# \### Task 1 — Methane Yield Prediction

# 

# \*\*Objective:\*\* predict experimental methane yield expressed as:

# 

# ```text

# Methane\_Yield\_mL\_CH4\_gVS

# ```

# 

# The deployed model is a \*\*tuned Artificial Neural Network / MLPRegressor\*\*.

# 

# The prediction pipeline includes:

# 

# \- substrate target encoding

# \- median imputation of selected numerical variables

# \- numerical feature scaling

# \- reactor-type encoding

# \- exact feature-order reconstruction

# \- ANN-based methane-yield prediction

# 

# The output is expressed as:

# 

# ```text

# mL CH4 / gVS

# ```

# 

# \---

# 

# \### Task 2 — Inhibition Risk Detection

# 

# \*\*Objective:\*\* determine whether the anaerobic-digestion process is likely to experience inhibition.

# 

# The binary target is represented as:

# 

# ```text

# 0 → No inhibition

# 1 → Inhibition

# ```

# 

# The deployed model is an \*\*Artificial Neural Network / MLP binary classifier\*\*.

# 

# AnaeroVision calculates an inhibition probability and converts it into a process-risk decision using the selected decision threshold:

# 

# ```text

# 0.14

# ```

# 

# If the calculated inhibition probability exceeds this threshold, the system activates Task 3.

# 

# \---

# 

# \### Task 3 — Inhibition Type Classification

# 

# \*\*Objective:\*\* identify the most probable inhibition mechanism when inhibition risk is detected.

# 

# The deployed model is a \*\*tuned Random Forest classifier\*\*.

# 

# The final inhibition categories include:

# 

# \- Ammonia inhibition

# \- Long-chain fatty acid (LCFA) inhibition

# \- Salinity inhibition

# \- Volatile fatty acid (VFA) accumulation

# \- Other inhibition

# 

# In addition to the predicted inhibition mechanism, AnaeroVision returns:

# 

# \- prediction confidence

# \- confidence level

# \- probability associated with each inhibition class

# 

# This provides additional diagnostic information instead of returning only a single categorical prediction.

# 

# \---

# 

# \## Model Performance

# 

# | Task | Final Model | Main Reported Performance |

# |---|---|---:|

# | Methane Yield Prediction | Tuned ANN / MLPRegressor | \*\*Test R² = 0.87\*\* |

# | Inhibition Risk Detection | ANN / MLP Classifier | \*\*Accuracy = 97.67%\*\* |

# | Inhibition Type Classification | Tuned Random Forest | \*\*OOF Accuracy = 92.73%\*\* |

# | Inhibition Type Classification | Tuned Random Forest | \*\*Macro F1 = 93.68%\*\* |

# 

# For inhibition-type classification, \*\*macro F1-score\*\* was particularly important because the inhibition classes were not perfectly balanced.

# 

# Macro F1 provides a more balanced assessment of the classifier's performance across the different inhibition mechanisms rather than allowing the most frequent classes to dominate the evaluation.

# 

# \---

# 

# \## Confidence-Based Inhibition Diagnosis

# 

# AnaeroVision does not interpret every inhibition-type prediction with the same level of certainty.

# 

# The final Task 3 model uses confidence thresholds to provide a more cautious interpretation.

# 

# | Prediction Confidence | Interpretation |

# |---:|---|

# | \*\*≥ 0.60\*\* | High confidence — likely inhibition type |

# | \*\*0.45–0.60\*\* | Moderate confidence — possible inhibition type; verify with process indicators |

# | \*\*0.40–0.45\*\* | Low confidence — tentative prediction |

# | \*\*< 0.40\*\* | Uncertain prediction |

# 

# This confidence-based interpretation reduces the risk of presenting uncertain predictions as definitive process diagnoses.

# 

# \---

# 

# \## Input Variables

# 

# AnaeroVision uses operational, feedstock, and reactor-related information describing the anaerobic-digestion system.

# 

# | Variable | Description |

# |---|---|

# | `Substrate` | Feedstock / substrate identity |

# | `Substrate\_Category` | General substrate category |

# | `Reactor\_Type` | Anaerobic digester configuration |

# | `Temperature\_C` | Operating temperature in °C |

# | `pH` | Digester pH |

# | `HRT\_days` | Hydraulic retention time |

# | `OLR\_gVS\_L\_day` | Organic loading rate |

# | `ISR` | Inoculum-to-substrate ratio |

# | `TAN\_mg\_L` | Total ammonia nitrogen concentration |

# | `VFA\_mg\_L` | Volatile fatty acid concentration |

# 

# These variables represent major aspects of anaerobic-digestion operation including:

# 

# \- substrate characteristics

# \- reactor configuration

# \- operating temperature

# \- acid-base conditions

# \- hydraulic retention

# \- organic loading

# \- inoculation conditions

# \- ammonia-related stress

# \- volatile fatty acid accumulation

# \- process stability

# 

# \---

# 

# \## Dataset

# 

# The machine-learning workflow was developed using an operational anaerobic-digestion dataset containing heterogeneous AD observations and operating conditions.

# 

# The data include information related to:

# 

# \- substrates

# \- reactor configurations

# \- operating conditions

# \- methane yield

# \- inhibition status

# \- inhibition mechanisms

# 

# Because the available operational dataset was relatively limited compared with the number of variables required for machine-learning development, a \*\*Gaussian Copula data-augmentation strategy\*\* was applied.

# 

# The final augmented dataset contained:

# 

# ```text

# 1,500 observations

# ```

# 

# The Gaussian Copula approach was used to generate synthetic observations while preserving the general statistical dependency structure of the cleaned operational dataset.

# 

# Synthetic augmentation was used to support:

# 

# \- model-training stability

# \- algorithm comparison

# \- broader representation of operational combinations

# 

# Synthetic observations were \*\*not intended to replace experimental evidence\*\*.

# 

# \---

# 

# \## Exploratory Data Analysis

# 

# Exploratory data analysis was performed before final model development to evaluate the structure and quality of the operational dataset.

# 

# The analysis included:

# 

# \- dataset structure inspection

# \- variable-type assessment

# \- missing-value analysis

# \- descriptive statistics

# \- Pearson correlation analysis

# \- methane-yield feature correlations

# \- IQR-based outlier detection

# \- numerical-variable distribution analysis

# \- categorical-variable inspection

# \- inhibition-class analysis

# 

# \### Correlation Analysis

# 

# Pearson correlation analysis was applied to the main numerical operating variables.

# 

# The analysis indicated that methane yield was not dominated by a single strongly correlated numerical variable.

# 

# This supports the use of nonlinear multivariable machine-learning approaches capable of learning interactions between multiple process parameters.

# 

# \### Outlier Analysis

# 

# Potential outliers were identified using the \*\*interquartile range (IQR)\*\* method for important variables such as:

# 

# \- TAN

# \- VFA

# \- OLR

# \- methane yield

# 

# Extreme observations were flagged and inspected rather than automatically removed.

# 

# In anaerobic-digestion systems, unusual operational values may represent genuine stress or inhibition conditions and can therefore contain useful process information.

# 

# \---

# 

# \## Data Preprocessing

# 

# The modelling workflow uses several preprocessing operations depending on the prediction task and machine-learning algorithm.

# 

# \### Numerical Preprocessing

# 

# The numerical preprocessing pipeline includes:

# 

# \- missing-value identification

# \- median imputation where required

# \- numerical standardization for scale-sensitive models

# \- preservation of correct feature ordering during deployment

# 

# \### Substrate Encoding

# 

# The substrate variable is categorical and can contain many different substrate descriptions.

# 

# For Tasks 1 and 2, substrate information is transformed using \*\*target encoding\*\*.

# 

# This allows substrate identity to contribute predictive information while avoiding a very high-dimensional one-hot encoded representation.

# 

# \### Reactor-Type Processing

# 

# Different reactor descriptions were consolidated into a smaller set of standardized categories.

# 

# Examples include:

# 

# ```text

# Batch

# CSTR

# Thermophilic Batch

# Other

# ```

# 

# The processed reactor category is then transformed into the representation expected by the deployed model.

# 

# \### Data-Leakage Control

# 

# Preprocessing operations such as:

# 

# \- encoding

# \- imputation

# \- scaling

# 

# were designed to be fitted using training data before being applied to validation or testing data.

# 

# This helps reduce information leakage during model evaluation.

# 

# \---

# 

# \## Machine-Learning Models Investigated

# 

# Multiple supervised machine-learning algorithms were evaluated during development.

# 

# Depending on the prediction task, the investigated models included:

# 

# \- Artificial Neural Networks

# \- Multilayer Perceptrons

# \- Random Forest

# \- XGBoost

# \- Support Vector Machine

# \- Support Vector Regression

# \- Logistic Regression

# 

# Rather than selecting one model family for the entire framework, the best-performing model was selected independently for each prediction task.

# 

# The final deployed combination is:

# 

# ```text

# Task 1

# Methane Yield Prediction

# &#x20;       ↓

# Tuned ANN / MLPRegressor

# 

# 

# Task 2

# Inhibition Risk Detection

# &#x20;       ↓

# ANN / MLP Classifier

# 

# 

# Task 3

# Inhibition Type Classification

# &#x20;       ↓

# Tuned Random Forest

# ```

# 

# This reflects the fact that different machine-learning algorithms may perform differently depending on whether the problem involves regression, binary classification, or multiclass classification.

# 

# \---

# 

# \## AnaeroVision Interface

# 

# AnaeroVision includes a graphical user interface developed using \*\*Streamlit\*\*.

# 

# The interface connects the three trained prediction modules into a single workflow.

# 

# Users enter one set of anaerobic-digestion operating conditions.

# 

# AnaeroVision then performs the required preprocessing and automatically executes the prediction cascade.

# 

# \### Methane Performance Output

# 

# The application returns:

# 

# ```text

# Predicted methane yield

# ```

# 

# expressed in:

# 

# ```text

# mL CH4 / gVS

# ```

# 

# \### Inhibition Risk Output

# 

# The application calculates:

# 

# ```text

# Inhibition probability

# ```

# 

# and reports whether an inhibition risk has been detected according to the selected decision threshold.

# 

# \### Inhibition Diagnosis

# 

# When inhibition is detected, AnaeroVision activates the inhibition-type classifier and reports:

# 

# ```text

# Predicted inhibition mechanism

# Prediction confidence

# Confidence level

# Class probabilities

# ```

# 

# The interface therefore transforms several independent machine-learning models into one connected process-intelligence workflow.

# 

# \---

# 

# \## Project Architecture

# 

# The core deployment structure of AnaeroVision is organized as follows:

# 

# ```text

# AnaeroVision/

# │

# ├── anaerovision3.py

# │

# ├── README.md

# ├── requirements.txt

# ├── LICENSE

# ├── .gitignore

# │

# ├── final\_task1\_methane\_yield\_model/

# │   │

# │   ├── ann\_tuned\_task1\_model.pkl

# │   ├── median\_imputer\_task1.pkl

# │   ├── standard\_scaler\_task1.pkl

# │   ├── target\_encoder\_task1.pkl

# │   ├── reactor\_mapping.json

# │   └── task1\_feature\_names.json

# │

# ├── final\_task2\_inhibition\_risk\_model/

# │   │

# │   ├── ann\_mlp\_task2\_final\_model.pkl

# │   ├── median\_imputer\_task2\_final.pkl

# │   ├── standard\_scaler\_task2\_final.pkl

# │   ├── target\_encoder\_task2\_final.pkl

# │   ├── reactor\_mapping\_task2\_final.json

# │   ├── task2\_final\_feature\_names.json

# │   └── task2\_final\_model\_summary.json

# │

# ├── final\_task3\_inhibition\_type\_model/

# │   │

# │   ├── task3\_final\_tuned\_rf\_pipeline.pkl

# │   ├── task3\_label\_mapping.json

# │   └── task3\_confidence\_thresholds.json

# │

# └── \[EDA and model-development scripts]

# ```

# 

# The three final model directories contain the trained models and the preprocessing objects required for inference.

# 

# \---

# 

# \## Installation

# 

# \### 1. Clone the Repository

# 

# Open Windows PowerShell and run:

# 

# ```powershell

# git clone https://github.com/Louay22-stack/AnaeroVision.git

# ```

# 

# Move into the repository:

# 

# ```powershell

# cd AnaeroVision

# ```

# 

# \---

# 

# \### 2. Create a Virtual Environment

# 

# Create a Python virtual environment:

# 

# ```powershell

# python -m venv .venv

# ```

# 

# Activate it:

# 

# ```powershell

# .\\.venv\\Scripts\\Activate.ps1

# ```

# 

# When activated, the terminal should display something similar to:

# 

# ```text

# (.venv) PS C:\\...\\AnaeroVision>

# ```

# 

# \---

# 

# \### 3. Install Dependencies

# 

# Install the dependencies from `requirements.txt`:

# 

# ```powershell

# pip install -r requirements.txt

# ```

# 

# The application primarily relies on packages such as:

# 

# ```text

# streamlit

# pandas

# numpy

# joblib

# scikit-learn

# category-encoders

# ```

# 

# Model-development and exploratory-analysis scripts may additionally use packages such as:

# 

# ```text

# matplotlib

# xgboost

# openpyxl

# ```

# 

# \---

# 

# \## Running AnaeroVision

# 

# From the root directory of the repository, run:

# 

# ```powershell

# streamlit run anaerovision3.py

# ```

# 

# Streamlit should launch the application and provide an address similar to:

# 

# ```text

# http://localhost:8501

# ```

# 

# Open this address in your browser.

# 

# \---

# 

# \## Example Prediction Workflow

# 

# A typical AnaeroVision prediction follows this sequence:

# 

# ```text

# 1\. Enter substrate information

# &#x20;             ↓

# 2\. Select reactor configuration

# &#x20;             ↓

# 3\. Enter operating conditions

# &#x20;             ↓

# 4\. Run AnaeroVision

# &#x20;             ↓

# 5\. Predict methane yield

# &#x20;             ↓

# 6\. Estimate inhibition probability

# &#x20;             ↓

# 7\. Compare probability with inhibition threshold

# &#x20;             ↓

# 8\. If no inhibition is detected:

# &#x20;     return methane prediction

# &#x20;     + no-risk classification

# &#x20;             ↓

# 9\. If inhibition is detected:

# &#x20;     activate Task 3

# &#x20;             ↓

# 10\. Predict inhibition mechanism

# &#x20;             ↓

# 11\. Report confidence and class probabilities

# ```

# 

# \---

# 

# \## Inhibition Mechanisms

# 

# The multiclass diagnostic model focuses on major inhibition mechanisms represented in the training dataset.

# 

# \### Ammonia Inhibition

# 

# Ammonia-related inhibition may occur when nitrogen-rich substrates lead to elevated ammonia concentrations.

# 

# AnaeroVision uses TAN together with other operational variables to help identify patterns consistent with ammonia-related inhibition.

# 

# \### LCFA Inhibition

# 

# Long-chain fatty acids may accumulate during the anaerobic degradation of lipid-rich substrates.

# 

# High LCFA concentrations can interfere with microbial activity and methane production.

# 

# \### VFA Accumulation

# 

# Volatile fatty acid accumulation may indicate an imbalance between acid-producing and methane-producing microbial populations.

# 

# Operational variables such as:

# 

# \- pH

# \- VFA

# \- OLR

# \- HRT

# 

# can provide important information related to acid accumulation.

# 

# \### Salinity Inhibition

# 

# Elevated salt concentrations can negatively affect anaerobic microbial communities and may contribute to reduced methane production.

# 

# \### Other Inhibition

# 

# Cases that cannot be assigned to one of the major predefined mechanisms are represented by the broader:

# 

# ```text

# Other inhibition

# ```

# 

# category.

# 

# \---

# 

# \## Scientific Interpretation

# 

# AnaeroVision is intended as a \*\*machine-learning decision-support framework\*\*.

# 

# It should not be interpreted as a replacement for:

# 

# \- laboratory measurements

# \- chemical analysis

# \- microbial analysis

# \- calibrated process sensors

# \- mechanistic anaerobic-digestion models

# \- professional process-engineering judgment

# 

# Model predictions should be considered together with:

# 

# \- reactor history

# \- operating trends

# \- substrate composition

# \- process measurements

# \- analytical results

# \- engineering knowledge

# 

# Particular caution is necessary when evaluating operating conditions that differ substantially from the conditions represented in the training dataset.

# 

# \---

# 

# \## Limitations

# 

# Several limitations should be considered when interpreting AnaeroVision predictions.

# 

# \### Dataset Heterogeneity

# 

# The underlying dataset combines anaerobic-digestion observations originating from heterogeneous experimental conditions.

# 

# Different studies may use different:

# 

# \- reactor configurations

# \- analytical techniques

# \- substrate definitions

# \- operating strategies

# \- measurement procedures

# 

# \### Synthetic Data

# 

# Gaussian Copula augmentation was used to increase the number of observations available for machine-learning development.

# 

# Synthetic observations can improve modelling stability but do not replace independently measured experimental data.

# 

# \### Missing Information

# 

# Some operational variables were incomplete and required preprocessing and imputation.

# 

# \### Class Imbalance

# 

# Inhibition cases were less frequent than non-inhibition cases.

# 

# Inhibition-type classes were also not equally represented.

# 

# \### Limited Inhibition Categories

# 

# Only inhibition mechanisms with sufficient usable data could be included in the supervised multiclass model.

# 

# \### External Validation

# 

# The models should be further evaluated using independent anaerobic-digestion datasets and experimentally generated reactor data.

# 

# \---

# 

# \## Future Development

# 

# Potential extensions of AnaeroVision include:

# 

# \- external validation using independent experimental datasets

# \- integration of real-time anaerobic-digester sensor data

# \- additional inhibition mechanisms

# \- uncertainty quantification

# \- explainable-AI methods

# \- SHAP-based feature interpretation

# \- time-series modelling

# \- early-warning systems

# \- process optimization

# \- continuous process monitoring

# \- automatic model retraining

# \- integration with industrial biogas plants

# \- integration with laboratory information systems

# \- API development

# \- cloud deployment

# \- digital-twin integration

# \- process-control integration

# 

# A future version of AnaeroVision could evolve from a static prediction interface toward a continuous \*\*AI-assisted anaerobic-digestion monitoring and decision-support platform\*\*.

# 

# \---

# 

# \## Research Context

# 

# AnaeroVision was developed within research focused on:

# 

# > \*\*AI-assisted optimization of anaerobic digestion for biogas and methane production\*\*

# 

# The project combines several disciplines:

# 

# ```text

# Anaerobic Digestion

# &#x20;       ×

# Biogas Engineering

# &#x20;       ×

# Bioprocess Engineering

# &#x20;       ×

# Environmental Biotechnology

# &#x20;       ×

# Machine Learning

# &#x20;       ×

# Artificial Intelligence

# &#x20;       ×

# Decision Support

# ```

# 

# The project was developed during a \*\*Mitacs Globalink Research Internship at Memorial University of Newfoundland, Canada\*\*.

# 

# \---

# 

# \## Technology Stack

# 

# AnaeroVision primarily uses:

# 

# | Technology | Purpose |

# |---|---|

# | Python | Core programming language |

# | Streamlit | User interface and deployment |

# | Pandas | Tabular data processing |

# | NumPy | Numerical operations |

# | Scikit-learn | Machine-learning models and preprocessing |

# | Joblib | Model serialization |

# | Category Encoders | Target encoding |

# | XGBoost | Gradient-boosting model evaluation |

# | Matplotlib | Scientific visualization |

# | OpenPyXL | Excel dataset handling |

# | Git | Version control |

# | GitHub | Source-code hosting |

# 

# \---

# 

# \## Repository Purpose

# 

# This repository provides the code and trained machine-learning components used to implement the AnaeroVision decision-support framework.

# 

# It is intended to support:

# 

# \- research transparency

# \- reproducibility

# \- further development

# \- machine-learning experimentation

# \- anaerobic-digestion modelling research

# \- educational use

# \- future external validation

# 

# \---

# 

# \## Author

# 

# \### Louay Ouhibi

# 

# Industrial Biology / Bioprocess Engineering

# 

# Research interests:

# 

# \- Anaerobic Digestion

# \- Biogas

# \- Biomethane

# \- Bioprocess Engineering

# \- Environmental Biotechnology

# \- Artificial Intelligence

# \- Machine Learning

# \- Sustainable Energy

# 

# GitHub:

# 

# \[github.com/Louay22-stack](https://github.com/Louay22-stack)

# 

# \---

# 

# \## Citation

# 

# If you use AnaeroVision, its code, or its modelling framework in academic or technical work, please cite the repository.

# 

# ```bibtex

# @software{ouhibi2026anaerovision,

# &#x20; author    = {Louay Ouhibi},

# &#x20; title     = {AnaeroVision: AI-Powered Anaerobic Digestion Process Intelligence},

# &#x20; year      = {2026},

# &#x20; publisher = {GitHub},

# &#x20; url       = {https://github.com/Louay22-stack/AnaeroVision}

# }

# ```

# 

# A formal publication citation can be added when the associated research manuscript becomes available.

# 

# \---

# 

# \## Acknowledgements

# 

# This project was developed in the context of the \*\*Mitacs Globalink Research Internship Program\*\* at \*\*Memorial University of Newfoundland (MUN), Canada\*\*.

# 

# The project benefited from the research environment and scientific support provided during the development of the anaerobic-digestion machine-learning workflow.

# 

# \---

# 

# \## Disclaimer

# 

# AnaeroVision is intended for \*\*research and decision-support purposes\*\*.

# 

# Predictions generated by the models should not be considered definitive biological or engineering diagnoses.

# 

# The software should not be used as the sole basis for operational changes in full-scale anaerobic-digestion facilities without appropriate experimental verification and professional evaluation.

# 

# \---

# 

# \## License

# 

# Please refer to the \[`LICENSE`](LICENSE) file included in this repository for licensing information.

# 

# \---

# 

# <div align="center">

# 

# \# AnaeroVision

# 

# \### Anaerobic Digestion × Bioprocess Engineering × Artificial Intelligence

# 

# \*\*From operating conditions to methane performance, process stability, and inhibition intelligence.\*\*

# 

# </div>

