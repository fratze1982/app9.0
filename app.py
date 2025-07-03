import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="KI-Vorhersage für Lackrezepturen", layout="wide")
st.title("🎨 KI-Vorhersage für Lackrezepturen mit Viskositätskurve")

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

# --- Erkennung von Viskositäts-Spalten anhand typischer Scherraten ---
visko_scherraten = ['0,1','0,209','0,436','1','1,9','3,28','10','17,3','36,2','53','100','329','687','1000','3010']
visko_scherraten_norm = [s.replace(',', '.') for s in visko_scherraten]
visko_spalten = [col for col in df.columns if col.replace(',', '.') in visko_scherraten_norm]

# --- Zielgrößen manuell wählbar ---
alle_spalten = df.columns.tolist()
st.markdown("---")
st.header("🎯 Zielgrößen & Rohstoffe definieren")

zielspalten = st.multiselect(
    "🎯 Wähle die Zielgrößen (Kennwerte)",
    options=[s for s in alle_spalten if s not in visko_spalten],
    default=[]
)

if not zielspalten:
    st.warning("Bitte mindestens eine Zielgröße auswählen.")
    st.stop()

X = df.drop(columns=zielspalten + visko_spalten, errors="ignore")
y = df[zielspalten].copy()

# Numerisch/kategorisch trennen
kategorisch = X.select_dtypes(include="object").columns.tolist()
numerisch = X.select_dtypes(include=[np.number]).columns.tolist()

# Encoding + Clean
X_encoded = pd.get_dummies(X)
df_encoded = X_encoded.copy()
df_encoded[y.columns] = y
df_encoded = df_encoded.dropna()
X_clean = df_encoded[X_encoded.columns]
y_clean = df_encoded[y.columns]

# --- Modelltraining ---
modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
modell.fit(X_clean, y_clean)

# --- Eingabeformular ---
st.sidebar.header("🛠 Parameter anpassen")
user_input = {}
for col in numerisch:
    user_input[col] = st.sidebar.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
for col in kategorisch:
    user_input[col] = st.sidebar.selectbox(col, sorted(df[col].dropna().unique()))

input_df = pd.DataFrame([user_input])
input_encoded = pd.get_dummies(input_df)
for col in X_clean.columns:
    if col not in input_encoded:
        input_encoded[col] = 0
input_encoded = input_encoded[X_clean.columns]

# --- Vorhersage ---
prediction = modell.predict(input_encoded)[0]
st.subheader("🔮 Vorhergesagte Zielgrößen")
for i, ziel in enumerate(zielspalten):
    st.metric(label=ziel, value=round(prediction[i], 2))

# --- Viskositätskurve (Originaldaten) ---
st.markdown("---")
st.header("💧 Viskositätskurve aus Datei")
if visko_spalten:
    try:
        visko_df = df[visko_spalten].copy()
        visko_df.columns = [float(c.replace(',', '.')) for c in visko_spalten]
        visko_df_plot = visko_df.T
        visko_df_plot.columns = [f"Messung {i+1}" for i in range(len(visko_df_plot.columns))]

        fig, ax = plt.subplots()
        visko_df_plot.plot(ax=ax, legend=False)
        ax.set_xlabel("Scherrate [1/s]")
        ax.set_ylabel("Viskosität [mPa·s]")
        ax.set_title("Gemessene Viskositätskurven")
        ax.set_xscale("log")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Fehler beim Zeichnen der Viskositätskurve: {e}")
else:
    st.info("ℹ️ Keine Viskositätskurvendaten gefunden.")

# --- Vorhersage der Viskositätskurve ---
if visko_spalten:
    st.header("🧠 KI-Vorhersage der Viskositätskurve")
    try:
        visko_y = df[visko_spalten].copy()
        visko_y.columns = [float(c.replace(',', '.')) for c in visko_spalten]

        visko_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
        visko_model.fit(X_clean, visko_y.loc[X_clean.index])

        visko_pred = visko_model.predict(input_encoded)[0]

        fig2, ax2 = plt.subplots()
        ax2.plot(visko_y.columns, visko_pred, marker="o")
        ax2.set_xlabel("Scherrate [1/s]")
        ax2.set_ylabel("Viskosität [mPa·s]")
        ax2.set_title("Vorhergesagte Viskositätskurve")
        ax2.set_xscale("log")
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"Fehler bei der Viskositätsvorhersage: {e}")
