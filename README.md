# 🌾 Crop Yield Prediction (Regression)

A beginner-friendly **Machine Learning** project that predicts the **yield of a crop
(tonnes per hectare)** from weather, soil and farming inputs, with a web app built using **Streamlit**.

## 1. Problem Statement
Farmers and planners want to estimate how much a field will produce before harvest, so they can
plan fertilizer, irrigation, storage and selling. We build a model that takes details such as
rainfall, temperature, soil nutrients and fertilizer and outputs the expected yield.

- **Type of ML:** Supervised learning -> **Regression** (the output is a continuous number)
- **Input (X):** 12 features  |  **Output (y):** `yield_tonnes_per_ha`

## 2. Dataset
`data/crop_data.csv` - 1500 farm plots, 13 columns.
The data is **synthetic** (created by `generate_dataset.py`). Yield is calculated from agriculture-style
rules (every crop has an ideal rainfall/temperature, fertilizer has diminishing returns, soil pH near 6.5
is best, irrigation helps) plus random noise. Results therefore show that the *method* works; they do not
predict real farms. You may replace the CSV with a real dataset (keep the column names).

| Column | Meaning |
|---|---|
| crop | Rice / Wheat / Maize / Cotton / Soybean |
| season | Kharif / Rabi / Summer |
| rainfall_mm | rainfall in the season |
| temperature_c | average temperature |
| humidity_percent | air humidity |
| soil_ph | soil acidity (7 = neutral) |
| nitrogen, phosphorus, potassium | soil nutrients (kg/ha) |
| fertilizer_kg_per_ha | fertilizer applied |
| pesticide_kg_per_ha | pesticide applied |
| irrigation | Yes / No |
| **yield_tonnes_per_ha** | **Target:** crop yield |

## 3. Methodology
1. **Load data** with pandas (the code regenerates the data automatically if the CSV is missing).
2. **Pre-processing:** StandardScaler for numbers, OneHotEncoder for text columns, inside a scikit-learn `Pipeline`.
3. **Train/Test split:** 80% / 20%.
4. **Algorithms compared:** Linear Regression, Decision Tree (depth 8), Random Forest (200 trees), Gradient Boosting.
5. **Evaluation:** MAE, RMSE, R² score and 5-fold Cross-Validation.
6. **Prediction:** the best model is used in the web app.

## 4. Results (on the included dataset)
| Model | MAE (t/ha) | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.522 | 0.667 | 0.732 |
| Decision Tree | 0.493 | 0.643 | 0.751 |
| Random Forest | 0.342 | 0.439 | 0.884 |
| **Gradient Boosting** | **0.331** | **0.426** | **0.890** |

**Why tree-based models win:** the relation between weather and yield is *not a straight line*
(too much or too little rain both hurt), and Linear Regression can only draw straight-line relations.
Random Forest and Gradient Boosting capture these curves.
Run `python train_model.py` to reproduce the numbers and save charts to `outputs/`.

## 5. Project Structure
```
├── app.py                # Streamlit web app (4 pages)
├── ml_pipeline.py        # All ML code: load, preprocess, train, evaluate
├── train_model.py        # Terminal run: prints results + saves charts to outputs/
├── generate_dataset.py   # Creates the synthetic dataset
├── data/crop_data.csv    # Dataset
├── outputs/              # Charts for the report
├── requirements.txt      # Libraries needed
└── README.md
```

## 6. Run on your computer
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## 7. Deploy FREE on Streamlit Community Cloud
1. Create a **new public repository** on GitHub (https://github.com/new), e.g. `crop-yield-prediction`.
2. Click **Add file -> Upload files**, drag in all files and folders, choose **Commit directly to the main branch**, click **Commit changes**.
   `app.py` and `requirements.txt` must be in the main (root) folder of the repo.
3. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
4. Select the repository, branch `main`, main file path `app.py`, then **Deploy**.

## 8. Limitations and Future Work
- Synthetic data: try a real dataset (e.g. Indian Crop Production data from Kaggle / data.gov.in).
- Add more factors (soil type, seed variety, pest attacks) and tune hyper-parameters.
- Predictions are estimates, not guarantees.
