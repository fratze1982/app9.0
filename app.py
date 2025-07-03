import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="KI-Vorhersage für Lackrezepturen", layout="wide")
st.title("🎨 KI-Vorhersage für Lackrezepturen")

# --- Datei-Upload ---
uploaded_file = st.file_uploader("📁 CSV-Datei hochladen (mit ; getrennt)", type=["csv"])
if uploaded_file is None:
    st.warning("Bitte lade eine CSV-Datei hoch.")
    st.stop()

# --- CSV einlesen ---
try:
    df = pd.read_csv(uploaded_file, sep=";", decimal=",", on_bad_lines='skip')
    st.success("✅ Datei erfolgreich geladen.")
except Exception as e:
    st.error(f"❌ Fehler beim Einlesen der Datei: {e}")
    st.stop()

st.write("🧾 Gefundene Spalten:", df.columns.tolist())

# --- Spaltenauswahl für Flexibilität ---
alle_spalten = df.columns.tolist()
anzahl_rohstoffe = st.number_input("🔧 Anzahl der Rohstoffspalten (Rest = Zielgrößen)", min_value=1, max_value=len(alle_spalten)-1, value=6)
rohstoff_spalten = alle_spalten[:anzahl_rohstoffe]
ziel_spalten_kandidaten = alle_spalten[anzahl_rohstoffe:]

zielspalten = st.multiselect(
    "🎯 Wähle die Zielgrößen (Kennwerte)",
    options=ziel_spalten_kandidaten,
    default=ziel_spalten_kandidaten[:1]
)

if not zielspalten:
    st.error("❌ Bitte mindestens eine Zielgröße auswählen.")
    st.stop()

X = df[rohstoff_spalten]
y = df[zielspalten].copy()

# Spaltentypen bestimmen
kategorisch = X.select_dtypes(include="object").columns.tolist()
numerisch = X.select_dtypes(exclude="object").columns.tolist()

# One-Hot-Encoding
X_encoded = pd.get_dummies(X)

# Fehlende Werte bereinigen
df_encoded = X_encoded.copy()
df_encoded[y.columns] = y
df_encoded = df_encoded.dropna()

X_clean = df_encoded[X_encoded.columns]
y_clean = df_encoded[y.columns]

if X_clean.empty or y_clean.empty:
    st.error("❌ Keine gültigen Daten zum Trainieren.")
    st.stop()

# --- Modelltraining ---
modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
modell.fit(X_clean, y_clean)

# --- Benutzer-Eingabeformular ---
st.sidebar.header("🔧 Parameter anpassen")
user_input = {}

for col in numerisch:
    try:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())

        if np.isnan(min_val) or np.isnan(max_val) or min_val == max_val:
            continue

        user_input[col] = st.sidebar.slider(
            col, min_value=min_val, max_value=max_val, value=mean_val
        )
    except Exception as e:
        st.sidebar.warning(f"{col} konnte nicht verarbeitet werden: {e}")
        continue

for col in kategorisch:
    options = sorted(df[col].dropna().unique())
    user_input[col] = st.sidebar.selectbox(col, options)

input_df = pd.DataFrame([user_input])
input_encoded = pd.get_dummies(input_df)

# Fehlende Spalten auffüllen
for col in X_clean.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[X_clean.columns]

# --- Vorhersage ---
prediction = modell.predict(input_encoded)[0]

st.subheader("🔮 Vorhergesagte Zielgrößen")
for i, ziel in enumerate(zielspalten):
    st.metric(label=ziel, value=round(prediction[i], 2))

# --- Viskositätskurve (optional) anzeigen ---
scherraten = [0.1, 0.209, 0.436, 1, 1.9, 3.28, 10, 17.3, 36.2, 53, 100, 329, 687, 1000, 3010]
visko_cols = [str(sr) for sr in scherraten if str(sr) in df.columns]

if visko_cols:
    st.subheader("📈 Viskositätskurven (Messwerte)")
    fig, ax = plt.subplots()
    for idx in range(min(5, len(df))):
        y_vals = df.loc[idx, visko_cols].values.astype(float)
        ax.plot(scherraten[:len(y_vals)], y_vals, label=f"Messung {idx+1}")
    ax.set_xlabel("Scherrate [1/s]")
    ax.set_ylabel("Viskosität [mPa·s]")
    ax.set_title("Viskosität über Scherrate")
    ax.legend()
    st.pyplot(fig)
