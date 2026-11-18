"""
generate_dataset.py
-------------------
Creates a SYNTHETIC (artificial) crop dataset of 1500 farm plots.

Why synthetic?  Real farm data is hard to get and often messy. Here the yield is
calculated from agronomy-style rules + random noise, so the ML models have realistic
patterns to learn:
  * each crop has its own base yield
  * rainfall and temperature have an OPTIMUM (too little or too much reduces yield)
  * fertilizer helps, but with diminishing returns
  * soil pH close to 6.5 is best, irrigation adds a bonus

You can replace data/crop_data.csv with a real dataset (keep the same column names).
Run:  python generate_dataset.py
"""
import numpy as np
import pandas as pd


def make_dataframe(n=1500, seed=42):
    """Builds the synthetic dataset and returns it as a DataFrame."""
    rng = np.random.default_rng(seed)

    # crop: (base yield t/ha, ideal rainfall mm, ideal temperature C)
    crops = {
        "Rice":    (4.5, 1200, 27),
        "Wheat":   (3.8, 600, 20),
        "Maize":   (5.0, 800, 25),
        "Cotton":  (2.0, 700, 28),
        "Soybean": (2.5, 750, 24),
    }
    crop = rng.choice(list(crops), n)
    base = np.array([crops[c][0] for c in crop])
    ideal_rain = np.array([crops[c][1] for c in crop])
    ideal_temp = np.array([crops[c][2] for c in crop])

    rainfall = np.clip(rng.normal(ideal_rain, 250), 150, 2000).round(0)      # mm per season
    temperature = np.clip(rng.normal(ideal_temp, 4), 10, 40).round(1)        # deg C
    humidity = np.clip(rng.normal(65, 12, n), 25, 95).round(0)               # percent
    soil_ph = np.clip(rng.normal(6.6, 0.7, n), 4.5, 8.5).round(1)
    nitrogen = np.clip(rng.normal(70, 25, n), 10, 140).round(0)              # kg/ha in soil
    phosphorus = np.clip(rng.normal(45, 15, n), 10, 90).round(0)
    potassium = np.clip(rng.normal(50, 15, n), 10, 100).round(0)
    fertilizer = np.clip(rng.normal(120, 50, n), 0, 250).round(0)            # kg/ha applied
    pesticide = np.clip(rng.normal(2.0, 1.0, n), 0, 5).round(1)              # kg/ha
    irrigation = rng.choice(["Yes", "No"], n, p=[0.6, 0.4])
    season = rng.choice(["Kharif", "Rabi", "Summer"], n, p=[0.45, 0.4, 0.15])

    # ---- hidden yield formula (each factor is a multiplier between ~0.3 and ~1.2) ----
    rain_f = np.exp(-((rainfall - ideal_rain) / 450) ** 2)            # bell curve around ideal
    temp_f = np.exp(-((temperature - ideal_temp) / 7) ** 2)
    fert_f = 0.6 + 0.4 * (1 - np.exp(-fertilizer / 90))               # diminishing returns
    ph_f = np.exp(-((soil_ph - 6.5) / 1.5) ** 2)
    npk_f = 0.8 + 0.2 * np.clip((nitrogen + phosphorus + potassium) / 165, 0, 1.3)
    pest_f = 0.9 + 0.1 * np.clip(pesticide / 2.5, 0, 1)
    irr_f = np.where(irrigation == "Yes", 1.10, 1.0)

    yield_t = base * (0.4 + 0.6 * rain_f) * (0.5 + 0.5 * temp_f) * fert_f * (0.7 + 0.3 * ph_f) \
              * npk_f * pest_f * irr_f * 1.25
    yield_t = np.clip(yield_t + rng.normal(0, 0.25, n), 0.2, None).round(2)

    return pd.DataFrame({
        "crop": crop, "season": season, "rainfall_mm": rainfall, "temperature_c": temperature,
        "humidity_percent": humidity, "soil_ph": soil_ph, "nitrogen": nitrogen,
        "phosphorus": phosphorus, "potassium": potassium, "fertilizer_kg_per_ha": fertilizer,
        "pesticide_kg_per_ha": pesticide, "irrigation": irrigation,
        "yield_tonnes_per_ha": yield_t,
    })


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df = make_dataframe()
    df.to_csv("data/crop_data.csv", index=False)
    print("Saved data/crop_data.csv ->", df.shape)
    print(df["yield_tonnes_per_ha"].describe().round(2))
