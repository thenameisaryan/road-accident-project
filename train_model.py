import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

print("🔄 Loading dataset...")
df = pd.read_csv("Road_Accident_Data.csv")

# 1. Clean Column Names & Target Label
df.columns = df.columns.str.strip()
target_col = "Accident_Severity" if "Accident_Severity" in df.columns else "Severity"

df = df.dropna(subset=[target_col])
df[target_col] = df[target_col].astype(str).str.strip().str.capitalize()
df[target_col] = df[target_col].replace({"Fetal": "Fatal"})

# Map ordinal target severity (0=Slight, 1=Serious, 2=Fatal)
severity_ordinal_map = {"Slight": 0.0, "Serious": 1.0, "Fatal": 2.0}
df = df[df[target_col].isin(severity_ordinal_map.keys())].copy()
df["Target_Ordinal"] = df[target_col].map(severity_ordinal_map)

# 2. Feature Engineering & Categories
if "Accident_Hour" not in df.columns and "Time" in df.columns:
    df["Accident_Hour"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.hour

df["Accident_Hour"] = pd.to_numeric(df["Accident_Hour"], errors="coerce").fillna(12).astype(int)

light_order = [
    "Daylight",
    "Darkness - lights lit",
    "Darkness - lighting unknown",
    "Darkness - lights unlit",
    "Darkness - no lighting"
]

weather_order = [
    "Fine no high winds",
    "Fine + high winds",
    "Other",
    "Raining no high winds",
    "Raining + high winds",
    "Snowing no high winds",
    "Snowing + high winds",
    "Fog or mist"
]

surface_order = [
    "Dry",
    "Wet or damp",
    "Snow",
    "Frost or ice",
    "Flood over 3cm. deep"
]

area_order = ["Urban", "Rural"]

cat_cols = ["Light_Conditions", "Weather_Conditions", "Road_Surface_Conditions", "Urban_or_Rural_Area"]

for col in cat_cols:
    df[col] = df[col].fillna("Unknown").astype(str).str.strip()

df["Speed_limit"] = pd.to_numeric(df["Speed_limit"], errors="coerce").fillna(30)
df["Number_of_Vehicles"] = pd.to_numeric(df["Number_of_Vehicles"], errors="coerce").fillna(2)

# Fit Ordinal Encoder
encoder = OrdinalEncoder(
    categories=[light_order, weather_order, surface_order, area_order],
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

encoded_cats = encoder.fit_transform(df[cat_cols])
encoded_cat_cols = [f"{c}_code" for c in cat_cols]
df_cats = pd.DataFrame(encoded_cats, columns=encoded_cat_cols, index=df.index)

feature_cols = ["Speed_limit", "Number_of_Vehicles", "Accident_Hour"] + encoded_cat_cols
X = pd.concat([df[["Speed_limit", "Number_of_Vehicles", "Accident_Hour"]], df_cats], axis=1)
y = df["Target_Ordinal"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=df[target_col]
)

# Enforce Monotonic Constraints (+1 forces higher features to always increase severity)
monotonic_cst = [1, 1, 0, 1, 1, 1, 1]

model = HistGradientBoostingRegressor(
    monotonic_cst=monotonic_cst,
    max_iter=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

print("⚡ Training Monotonic Ordinal Model...")
model.fit(X_train, y_train)

# 4. Calculate Dynamic Calibration Percentiles (s_min and s_max)
train_preds = model.predict(X_train)
s_min = float(np.percentile(train_preds, 5))
s_max = float(np.percentile(train_preds, 95))

print(f"📊 Calibration Range -> Baseline Minimum (s_min): {s_min:.4f} | Baseline Maximum (s_max): {s_max:.4f}")

# 5. Save Artifacts
package = {
    "model": model,
    "encoder": encoder,
    "s_min": s_min,
    "s_max": s_max,
    "feature_cols": feature_cols,
    "cat_cols": cat_cols,
    "categories": {
        "light_order": light_order,
        "weather_order": weather_order,
        "surface_order": surface_order,
        "area_order": area_order
    }
}

joblib.dump(package, "accident_severity_model.pkl")
print("🎉 Model trained and calibrated successfully! Saved to 'accident_severity_model.pkl'.")
