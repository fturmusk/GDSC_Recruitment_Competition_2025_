GDSC Recruitment Competition 2025


🧩 Overview

This project is part of the GDSC Recruitment Competition 2025, a machine learning challenge organized by GDSC‑NSUT. The task consists of predicting the CORRUCYSTIC_DENSITY of alien‑like bio‑computational artifacts called CORRUCYSTS, discovered in a fictional future scenario. This storyline originates from the official competition description on Kaggle.

The dataset contains both training and test samples with numerical features.
Your goal is to train a regression model that accurately reconstructs the density value for unseen test samples.


The objective of this exercise is to reconstruct the "CORRUCYSTIC_DENSITY" feature of the test sample using the existing data.

🔗 Competition link:
https://www.kaggle.com/competitions/recruitment-task-for-gdsc-ml/overview


🗂️ Project Structure

```
GDSC_Recruitment_Competition_2025/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── src/
│   ├── preprocess.py        # Loading, cleaning, engineering, scaling
│   ├── exploration.py       # Exploratory data analysis
│   ├── model.py             # Model training + evaluation
│   └── visualisation.py     # Additional plots
│
├── main.py                  # Central execution pipeline
├── requirements.txt         # Dependencies
└── README.md
```


⚙️ Installation


git clone https://github.com/fturmusk/GDSC_Recruitment_Competition_2025_.git
cd GDSC_Recruitment_Competition_2025_

pip install -r requirements.txt

▶️ Usage

python main.py