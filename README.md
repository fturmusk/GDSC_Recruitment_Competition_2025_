GDSC Recruitment Competition 2025


🧩 Overview

This project is part of the GDSC Recruitment Competition 2025, a machine learning challenge organized by GDSC‑NSUT. The task consists of predicting the CORRUCYSTIC_DENSITY of alien‑like bio‑computational artifacts called CORRUCYSTS, discovered in a fictional future scenario. This storyline originates from the official competition description on Kaggle.

The dataset contains both training and test samples with numerical features.
Your goal is to train a regression model that accurately reconstructs the density value for unseen test samples.


The objective of this exercise is to reconstruct the "CORRUCYSTIC_DENSITY" feature of the test sample using the existing data.

🔗 Competition link:
https://www.kaggle.com/competitions/recruitment-task-for-gdsc-ml/overview

Projektstruktur

=> data/                 # CSV-Datei

=>src/
  ---preprocess.py     # Daten laden, Feature Engineering, Scaling
  
  ---model.py          # Training der Modelle
  
  ---exploration.py    # visualisation der Dataset
  
  ---visualisation.py            # Optional: 

=>README.md                 

=>main.py               # Hauptskript zur Ausführung
