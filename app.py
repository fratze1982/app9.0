import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="KI-Vorhersage für Lackrezepturen", layout="wide")
st.title("🎨 KI-Vorhersage für Lackrezepturen")

uploaded_file = st.file_uploader("📁 CSV-Datei hochladen", type=["csv"])
if uploaded_file is None:
    st.warning("Bitte lade eine CSV-Datei hoch.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file, sep=";", decimal=",", error_bad_lines=False)
    st.success("✅ Datei erfolgreich geladen.")
except Exception as e:
    st.error(f"❌ Fehler beim Einlesen der Datei: {e}")
    st.stop()

st.write("🧾 Gefundene Spalten:", df.columns.tolist())

alle_spalten = df.columns.tolist()
rohstoff_spalten = st.multiselect("🧪 Welche Spalten sind Rohstoffe (Einflussgrößen)?", options=alle_spalten, default=alle_spalten[:6])
zielspalten = st.multiselect("🎯 Wähle die Zielgrößen (Kennwerte)", options=[s for s in alle_spalten if s not in rohstoff_spalten], default=[alle_spalten[6]] if len(alle_spalten) > 6 else [])

if not rohstoff_spalten or not zielspalten:
    st.warning("Bitte wähle sowohl Rohstoffe als auch Zielgrößen aus.")
    st.stop()

X = df[rohstoff_spalten].copy()
y = df[zielspalten].copy()

X_encoded = pd.get_dummies(X)
df_encoded = X_encoded.copy()
df_encoded[y.columns] = y
df_encoded = df_encoded.dropna()

X_clean = df_encoded[X_encoded.columns]
y_clean = df_encoded[y.columns]

if X_clean.empty or y_clean.empty:
    st.error("❌ Keine gültigen Daten zum Trainieren.")
    st.stop()

modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
modell.fit(X_clean, y_clean)

st.sidebar.header("🔧 Eingabewerte anpassen")
user_input = {}
for col in rohstoff_spalten:
    try:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        user_input[col] = st.sidebar.slider(col, min_val, max_val, mean_val)
    except:
        continue

input_df = pd.DataFrame([user_input])
input_encoded = pd.get_dummies(input_df)
for col in X_clean.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[X_clean.columns]

prediction = modell.predict(input_encoded)[0]

st.subheader("🔮 Vorhergesagte Zielgrößen")
for i, ziel in enumerate(zielspalten):
    st.metric(label=ziel, value=round(prediction[i], 2))

# --- Viskositätskurve plotten (optional) ---
scherraten_kandidaten = [
    "0.1", "0.209", "0.436", "1", "1.9", "3.28", "10", "17.3",
    "36.2", "53", "100", "329", "687", "1000", "3010"
]

gemessene_scherraten = [col for col in zielspalten if col in scherraten_kandidaten]

if len(gemessene_scherraten) >= 3:
    st.subheader("🧪 Vorhergesagte Viskositätskurve")
    if st.checkbox("📈 Viskositätskurve anzeigen"):
        x = [float(sr) for sr in gemessene_scherraten]
        y_vals = [prediction[zielspalten.index(s)] for s in gemessene_scherraten]

        fig, ax = plt.subplots()
        ax.plot(x, y_vals, marker="o")
        ax.set_xscale("log")
        ax.set_xlabel("Scherrate [1/s]")
        ax.set_ylabel("Viskosität [mPa·s]")
        ax.set_title("Vorhergesagte Viskositätskurve")
        st.pyplot(fig)
