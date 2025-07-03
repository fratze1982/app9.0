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

# --- Spaltenwahl für Rohstoffe und Zielgrößen ---
alle_spalten = df.columns.tolist()
st.sidebar.subheader("📌 Spaltenkonfiguration")
rohstoff_spalten = st.sidebar.multiselect("Rohstoffe auswählen", options=alle_spalten, default=alle_spalten[:6])
zielspalten = st.sidebar.multiselect("🎯 Zielgrößen auswählen", options=[s for s in alle_spalten if s not in rohstoff_spalten], default=[alle_spalten[6]] if len(alle_spalten) > 6 else [])

if not zielspalten:
    st.warning("Bitte wähle mindestens eine Zielgröße aus.")
    st.stop()

X = df[rohstoff_spalten].copy()
y = df[zielspalten].copy()

# --- Modelltraining ---
modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
modell.fit(X, y)

# --- Benutzer-Eingabeformular ---
st.sidebar.header("🛠️ Rohstoff-Eingaben")
user_input = {}
for col in rohstoff_spalten:
    try:
        user_input[col] = st.sidebar.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    except:
        continue

input_df = pd.DataFrame([user_input])
prediction = modell.predict(input_df)[0]

st.subheader("🔮 Vorhergesagte Zielgrößen")
for i, ziel in enumerate(zielspalten):
    st.metric(label=ziel, value=round(prediction[i], 2))

# --- Zieloptimierung ---
st.subheader("🎯 Zieloptimierung (Zufallssuche)")
st.sidebar.subheader("⚙️ Zielvorgaben und Toleranzen")
zielwerte = {}
toleranzen = {}
gewichtung = {}
for ziel in zielspalten:
    zielwerte[ziel] = st.sidebar.number_input(f"Zielwert für {ziel}", value=float(df[ziel].mean()))
    toleranzen[ziel] = st.sidebar.number_input(f"Toleranz (±) für {ziel}", value=2.0)
    gewichtung[ziel] = st.sidebar.slider(f"Gewichtung für {ziel}", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

st.sidebar.subheader("🔒 Rohstoffe fixieren / begrenzen")
fixierte_werte = {}
rohstoffgrenzen = {}
for roh in rohstoff_spalten:
    if st.sidebar.checkbox(f"{roh} fixieren?"):
        fixierte_werte[roh] = st.sidebar.number_input(f"Fixwert für {roh}", value=float(df[roh].mean()))
    else:
        min_val = float(df[roh].min())
        max_val = float(df[roh].max())
        if min_val == max_val:
            min_val -= 0.01
            max_val += 0.01
        rohstoffgrenzen[roh] = st.sidebar.slider(f"Grenzen für {roh}", min_val, max_val, (min_val, max_val))

# --- Simulation ---
anzahl = 1000
simulierter_input = []
scores = []
if st.button("🚀 Zielsuche starten"):
    for _ in range(anzahl):
        werte = {}
        for roh in rohstoff_spalten:
            if roh in fixierte_werte:
                werte[roh] = fixierte_werte[roh]
            else:
                werte[roh] = np.random.uniform(*rohstoffgrenzen[roh])
        simulierter_input.append(werte)
    sim_df = pd.DataFrame(simulierter_input)
    y_pred = modell.predict(sim_df)

    treffer, scores = [], []
    for i, row in enumerate(y_pred):
        passt = True
        score = 0
        for ziel in zielspalten:
            delta = abs(row[zielspalten.index(ziel)] - zielwerte[ziel])
            score += delta * gewichtung[ziel]
            if delta > toleranzen[ziel]:
                passt = False
        if passt:
            treffer.append(i)
            scores.append(score)

    if treffer:
        ergebnisse = sim_df.iloc[treffer].copy()
        ziel_df = pd.DataFrame([y_pred[i] for i in treffer], columns=zielspalten)
        ergebnis_df = pd.concat([ergebnisse.reset_index(drop=True), ziel_df.reset_index(drop=True)], axis=1)
        ergebnis_df.insert(0, "Score", [round(s, 2) for s in scores])

        st.success(f"✅ {len(ergebnis_df)} passende Formulierungen gefunden.")
        st.dataframe(ergebnis_df)

        csv = ergebnis_df.to_csv(index=False).encode("utf-8")
        st.download_button("📄 Ergebnisse als CSV herunterladen", data=csv, file_name="formulierungen.csv")

        st.subheader("📈 Zielgrößen der Top 10")
        top10 = ergebnis_df.head(10)
        fig, ax = plt.subplots()
        for ziel in zielspalten:
            ax.plot(top10["Score"], top10[ziel], marker="o", label=ziel)
        ax.set_xlabel("Score")
        ax.set_ylabel("Zielwerte")
        ax.legend()
        st.pyplot(fig)

        st.subheader("🔬 Radar-Diagramm der Top 3")
        if len(ergebnis_df) >= 3:
            radar_data = ergebnis_df.head(3)[zielspalten].copy()
            labels = zielspalten + [zielspalten[0]]
            angles = np.linspace(0, 2 * np.pi, len(zielspalten), endpoint=False).tolist()
            angles += angles[:1]
            fig, ax = plt.subplots(subplot_kw=dict(polar=True))
            for idx, row in radar_data.iterrows():
                values = row.tolist() + [row.tolist()[0]]
                ax.plot(angles, values, label=f"F{idx+1}")
                ax.fill(angles, values, alpha=0.1)
            ax.set_thetagrids(np.degrees(angles), labels)
            ax.set_title("Radarvergleich Zielgrößen")
            ax.legend()
            st.pyplot(fig)

        st.subheader("📊 Balkendiagramm ausgewählter Formulierungen")
        index_auswahl = st.multiselect("Formulierungen auswählen (max. 5)", options=list(range(len(ergebnis_df))), default=list(range(min(3, len(ergebnis_df)))))
        if index_auswahl:
            auswahl_df = ergebnis_df.loc[index_auswahl, zielspalten]
            auswahl_df["Formulierung"] = [f"F{i+1}" for i in index_auswahl]
            auswahl_df = auswahl_df.set_index("Formulierung")
            fig2, ax2 = plt.subplots()
            auswahl_df.plot(kind="bar", ax=ax2)
            ax2.set_title("Zielgrößenvergleich ausgewählter Formulierungen")
            st.pyplot(fig2)

# --- Viskositätskurve darstellen ---
scherwerte = [0.1, 0.209, 0.436, 1, 1.9, 3.28, 10, 17.3, 36.2, 53, 100, 329, 687, 1000, 3010]
scher_spalten = [s for s in df.columns if any(str(sw) in s for sw in scherwerte)]
if scher_spalten:
    st.subheader("🧪 Viskositätskurve")
    mittelwerte = df[scher_spalten].mean()
    fig3, ax3 = plt.subplots()
    ax3.plot(scherwerte[:len(mittelwerte)], mittelwerte, marker="o")
    ax3.set_xlabel("Scherrate")
    ax3.set_ylabel("Viskosität")
    ax3.set_title("Mittlere Viskositätskurve")
    st.pyplot(fig3)
