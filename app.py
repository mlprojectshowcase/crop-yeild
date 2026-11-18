"""
app.py  -  Streamlit web app for "Crop Yield Prediction" (Regression)
Run locally:  streamlit run app.py
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import ml_pipeline as ml

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="wide")


@st.cache_data
def get_data():
    return ml.load_data()


@st.cache_resource
def get_models():
    """Trains once when the app starts (a few seconds), then reuses the result."""
    return ml.train_and_evaluate(get_data())


df = get_data()
fitted, results, best, (_, X_test, _, y_test) = get_models()

st.sidebar.title("🌾 Navigation")
page = st.sidebar.radio("Go to", ["1. Project Overview", "2. Dataset Explorer",
                                  "3. Model Results", "4. Predict Crop Yield"])

# ------------------------------------------------------------------ PAGE 1
if page.startswith("1"):
    st.title("🌾 Crop Yield Prediction")
    st.write("""
**Goal:** predict how much crop a farm plot will produce (in **tonnes per hectare**) from
weather, soil and farming practices.

**Type of problem:** *Supervised regression* - the output is a **number** (not a category).
The computer learns from past plots whose yield is known, then predicts the yield of a new plot.

**Workflow used in this project**
1. Create / load the dataset
2. Pre-process the data (scale numbers, convert text to numbers)
3. Split into 80 % training and 20 % testing data
4. Train four algorithms: Linear Regression, Decision Tree, Random Forest, Gradient Boosting
5. Compare them using MAE, RMSE and R² score
6. Use the best model to predict new plots
""")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Farm plots", len(df))
    c2.metric("Input features", len(ml.NUMERIC) + len(ml.CATEGORICAL))
    c3.metric("Best model", best)
    c4.metric("R² of best model", f"{results[best]['r2']:.3f}")
    st.info("Note: the dataset is synthetic (artificially generated using agriculture-style rules "
            "plus random noise). See the README for details.")

# ------------------------------------------------------------------ PAGE 2
elif page.startswith("2"):
    st.title("📊 Dataset Explorer")
    st.subheader("First 10 rows")
    st.dataframe(df.head(10), use_container_width=True)
    st.subheader("Summary statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Average yield per crop")
        st.bar_chart(df.groupby("crop")[ml.TARGET].mean().sort_values())
    with col2:
        st.subheader("Feature vs yield")
        feat = st.selectbox("Choose a feature", ml.NUMERIC)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.scatter(df[feat], df[ml.TARGET], alpha=0.3, s=10)
        ax.set_xlabel(feat); ax.set_ylabel("Yield (t/ha)")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Note: yields differ by crop, so the cloud of points is spread out.")

    st.subheader("Correlation between numeric columns")
    cols = ml.NUMERIC + [ml.TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=8)
    fig.colorbar(im); fig.tight_layout()
    st.pyplot(fig)

# ------------------------------------------------------------------ PAGE 3
elif page.startswith("3"):
    st.title("🏆 Model Results")
    table = pd.DataFrame({
        n: {"MAE (t/ha)": r["mae"], "RMSE (t/ha)": r["rmse"], "R² score": r["r2"],
            "5-fold CV R²": r["cv_r2"]}
        for n, r in results.items()
    }).T.round(3)
    st.dataframe(table, use_container_width=True)
    st.success(f"Best model on the test data: **{best}** "
               f"(R² = {results[best]['r2']:.3f}, average error ≈ {results[best]['mae']:.2f} t/ha)")

    with st.expander("What do these metrics mean?"):
        st.write("""
- **MAE (Mean Absolute Error)** - on average, how many tonnes/hectare the prediction is off. *Lower is better.*
- **RMSE** - like MAE but punishes big mistakes more. *Lower is better.*
- **R² score** - how much of the ups and downs of yield the model explains. 1.0 = perfect, 0 = no better than guessing the average. *Higher is better.*
- **Cross-validation** - training/testing repeated 5 times on different parts of the data to check the result is stable.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Actual vs predicted ({best})")
        pred = results[best]["pred"]
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.scatter(y_test, pred, alpha=0.5)
        lim = [0, max(y_test.max(), pred.max()) + 0.5]
        ax.plot(lim, lim, "r--", label="Perfect prediction")
        ax.set_xlabel("Actual yield (t/ha)"); ax.set_ylabel("Predicted yield (t/ha)"); ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Points close to the red line = accurate predictions.")
    with col2:
        st.subheader("Feature importance (Random Forest)")
        st.bar_chart(ml.feature_importance(fitted["Random Forest"]))
        st.caption("Longer bar = that input influences the yield more.")

# ------------------------------------------------------------------ PAGE 4
else:
    st.title("🔮 Predict Crop Yield")
    st.write(f"Enter the plot details. Prediction is made by the best model: **{best}**.")
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            crop = st.selectbox("Crop", sorted(df["crop"].unique()))
            season = st.selectbox("Season", ["Kharif", "Rabi", "Summer"])
            irrigation = st.selectbox("Irrigation available", ["Yes", "No"])
            rain = st.slider("Rainfall (mm per season)", 150, 2000, 800, 10)
            temp = st.slider("Average temperature (°C)", 10.0, 40.0, 25.0, 0.5)
        with c2:
            hum = st.slider("Humidity (%)", 25, 95, 65)
            ph = st.slider("Soil pH", 4.5, 8.5, 6.5, 0.1)
            fert = st.slider("Fertilizer (kg/ha)", 0, 250, 120)
            pest = st.slider("Pesticide (kg/ha)", 0.0, 5.0, 2.0, 0.1)
        with c3:
            n = st.slider("Soil nitrogen (kg/ha)", 10, 140, 70)
            p = st.slider("Soil phosphorus (kg/ha)", 10, 90, 45)
            k = st.slider("Soil potassium (kg/ha)", 10, 100, 50)
        go = st.form_submit_button("Predict yield")

    if go:
        row = pd.DataFrame([{
            "crop": crop, "season": season, "rainfall_mm": rain, "temperature_c": temp,
            "humidity_percent": hum, "soil_ph": ph, "nitrogen": n, "phosphorus": p,
            "potassium": k, "fertilizer_kg_per_ha": fert, "pesticide_kg_per_ha": pest,
            "irrigation": irrigation}])
        value = float(fitted[best].predict(row)[0])
        avg = df.loc[df["crop"] == crop, ml.TARGET].mean()
        st.success(f"Predicted yield for {crop}: **{value:.2f} tonnes per hectare**")
        st.write(f"Typical average for {crop} in the dataset: {avg:.2f} t/ha "
                 f"({'above' if value >= avg else 'below'} average).")
        st.caption(f"The model's typical error is about ±{results[best]['mae']:.2f} t/ha.")
