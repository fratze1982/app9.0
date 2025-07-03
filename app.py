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

# --- Zielspaltenauswahl ---
numerische_spalten = df.select_dtypes(include=[np.number]).columns.tolist()
zielspalten = st.multiselect("🎯 Wähle die Zielgrößen (Kennwerte)", options=numerische_spalten, default=numerische_spalten[:1])

if not zielspalten:
    st.error("❌ Bitte mindestens eine Zielgröße auswählen.")
    st.stop()

X = df.drop(columns=zielspalten)
y = df[zielspalten].copy()

# --- Spaltentypen analysieren ---
kategorisch = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerisch = X.select_dtypes(include=[np.number]).columns.tolist()

# --- Eingabemaske mit robustem Slider ---
st.sidebar.header("🔧 Parameter anpassen")
user_input = {}
for col in numerisch:
    col_values = df[col].dropna()
    if col_values.empty:
        # Spalte überspringen, wenn keine Daten vorhanden sind
        continue

    min_val = float(col_values.min())
    max_val = float(col_values.max())
    mean_val = float(col_values.mean())

    # Falls min == max, kleinen Bereich erzeugen, damit Slider funktioniert
    if min_val == max_val:
        min_val -= 0.01
        max_val += 0.01

    # Mittelwert sicher innerhalb des Bereichs setzen
    if mean_val < min_val or mean_val > max_val:
        mean_val = (min_val + max_val) / 2

    # Optional Debug-Ausgabe
    # st.sidebar.write(f"{col}: min={min_val}, max={max_val}, mean={mean_val}")

    user_input[col] = st.sidebar.slider(col, min_val, max_val, mean_val)

for col in kategorisch:
    user_input[col] = st.sidebar.selectbox(col, df[col].dropna().unique())

input_df = pd.DataFrame([user_input])
X_encoded = pd.get_dummies(X)
input_encoded = pd.get_dummies(input_df)

# Fehlende Spalten in input_encoded ergänzen, um gleiche Struktur zu garantieren
for col in X_encoded.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[X_encoded.columns]

# --- Modelltraining ---
df_encoded = X_encoded.copy()
df_encoded[y.columns] = y
df_encoded = df_encoded.dropna()
X_clean = df_encoded[X_encoded.columns]
y_clean = df_encoded[y.columns]

modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
modell.fit(X_clean, y_clean)

# --- Vorhersage ---
prediction = modell.predict(input_encoded)[0]
st.subheader("🔮 Vorhergesagte Zielgrößen")
for i, ziel in enumerate(zielspalten):
    st.metric(label=ziel, value=round(prediction[i], 2))

# --- Zieloptimierung (Zufallsansatz) ---
st.subheader("🎯 Zieloptimierung per Zufallssuche")
zielwerte = {}
toleranzen = {}
gewichtung = {}

with st.expander("⚙️ Zielvorgaben & Toleranzen setzen"):
    for ziel in zielspalten:
        zielwerte[ziel] = st.number_input(f"Zielwert für {ziel}", value=float(df[ziel].mean()))
        toleranzen[ziel] = st.number_input(f"Toleranz für {ziel} (±)", value=2.0)
        gewichtung[ziel] = st.slider(f"Gewichtung für {ziel}", 0.0, 5.0, 1.0, 0.1)

steuerbare_rohstoffe = numerisch
fixierte_werte = {}
rohstoffgrenzen = {}

st.sidebar.header("🧪 Rohstoffe fixieren oder begrenzen")
for roh in steuerbare_rohstoffe:
    if st.sidebar.checkbox(f"{roh} fixieren?"):
        fixierte_werte[roh] = st.sidebar.number_input(f"Fixwert für {roh}", value=float(df[roh].mean()))
    else:
        col_values = df[roh].dropna()
        min_val = float(col_values.min())
        max_val = float(col_values.max())
        if min_val == max_val:
            min_val -= 0.01
            max_val += 0.01
        rohstoffgrenzen[roh] = st.sidebar.slider(f"Grenzen für {roh}", min_val, max_val, (min_val, max_val))

if st.button("🚀 Zielsuche starten"):
    sim_daten = []
    for _ in range(1000):
        rohwerte = {}
        for roh in steuerbare_rohstoffe:
            if roh in fixierte_werte:
                rohwerte[roh] = fixierte_werte[roh]
            else:
                rohwerte[roh] = np.random.uniform(*rohstoffgrenzen[roh])
        sim_daten.append(rohwerte)

    sim_df = pd.DataFrame(sim_daten)
    sim_encoded = pd.get_dummies(sim_df)
    for col in X_clean.columns:
        if col not in sim_encoded.columns:
            sim_encoded[col] = 0
    sim_encoded = sim_encoded[X_clean.columns]
    y_pred = modell.predict(sim_encoded)

    treffer = []
    for i, yhat in enumerate(y_pred):
        score = 0
        passt = True
        for ziel in zielspalten:
            delta = abs(yhat[zielspalten.index(ziel)] - zielwerte[ziel])
            score += delta * gewichtung[ziel]
            if delta > toleranzen[ziel]:
                passt = False
        if passt:
            treffer.append((i, score))

    if treffer:
        treffer.sort(key=lambda x: x[1])
        top_idx = [i for i, _ in treffer[:10]]
        ergebnis_df = pd.concat(
            [sim_df.iloc[top_idx].reset_index(drop=True),
             pd.DataFrame(y_pred[top_idx], columns=zielspalten)],
            axis=1
        )
        ergebnis_df.insert(0, "Score", [round(s, 2) for _, s in treffer[:10]])

        st.success(f"✅ {len(ergebnis_df)} passende Formulierungen gefunden.")
        st.dataframe(ergebnis_df)

        csv = ergebnis_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Ergebnisse herunterladen", data=csv, file_name="optimierte_formulierungen.csv")

        # --- Balkendiagramm ---
        st.subheader("📊 Vergleich Zielgrößen (Top 5)")
        vergleich_df = ergebnis_df.head(5).copy()
        fig, ax = plt.subplots(figsize=(10, 5))
        for ziel in zielspalten:
            ax.plot(vergleich_df["Score"], vergleich_df[ziel], label=ziel, marker="o")
        ax.set_xlabel("Score")
        ax.set_ylabel("Zielwert")
        ax.legend()
        st.pyplot(fig)

        # --- Radar-Diagramm ---
        st.subheader("🔬 Radar-Diagramm der Top 3")
        if len(ergebnis_df) >= 3:
            radar_df = ergebnis_df.head(3)[zielspalten].copy()
            labels = list(zielspalten)
            num_vars = len(labels)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]
            labels += labels[:1]

            fig, ax = plt.subplots(subplot_kw=dict(polar=True))
            for idx, row in radar_df.iterrows():
                values = row.tolist()
                values += values[:1]
                ax.plot(angles, values, label=f"Formulierung {idx+1}")
                ax.fill(angles, values, alpha=0.1)
            ax.set_thetagrids(np.degrees(angles), labels)
            ax.set_title("Radarvergleich Zielgrößen")
            ax.legend(loc="upper right")
            st.pyplot(fig)
    else:
        st.error("❌ Keine passenden Formulierungen innerhalb der Toleranzen gefunden.")
