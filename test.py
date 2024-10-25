import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# Charger le DataFrame depuis un fichier Excel ou créer un nouveau DataFrame si le fichier n'existe pas
try:
    df = pd.read_excel("base_de_donnees.xlsx")
except FileNotFoundError:
    # Créer un dictionnaire avec des données fictives
    data = {
        "marker_id": [1, 2, 3, 4, 5],
        "bloqué": [1, 1, 0, 1, 0],
        "forcé": [0, 1, 1, 0, 0],
        "latitude": [48.8566, 48.8570, 48.8575, 48.8580, 48.8585],
        "longitude": [2.3522, 2.3525, 2.3530, 2.3535, 2.3540],
    }
    df = pd.DataFrame(data)


# Fonction pour mettre à jour les colonnes "bloqué" et "forcé" d'un nœud
def update_node(node_id, blocked, forced):
    if node_id in df["marker_id"].values:
        # Mettre à jour les colonnes
        df.loc[df["marker_id"] == node_id, ["bloqué", "forcé"]] = blocked, forced
        df.to_excel("base_de_donnees.xlsx", index=False)


# Fonction pour réinitialiser les nœuds
def reset_node(node_id, reset_bloque, reset_force):
    if node_id in df["marker_id"].values:
        if reset_bloque:
            df.loc[df["marker_id"] == node_id, "bloqué"] = 0
        if reset_force:
            df.loc[df["marker_id"] == node_id, "forcé"] = 0
        df.to_excel("base_de_donnees.xlsx", index=False)


# Initialiser le state si non défini
if "data_changed" not in st.session_state:
    st.session_state.data_changed = False

# Afficher le DataFrame dans Streamlit
st.write("### État des nœuds")
st.dataframe(df)

# Ajouter un bouton pour réinitialiser tous les nœuds
if st.button("Réinitialiser tous les nœuds"):
    df["bloqué"] = 0
    df["forcé"] = 0
    df.to_excel("base_de_donnees.xlsx", index=False)
    st.success("Tous les nœuds ont été réinitialisés.")
    st.session_state.data_changed = True  # Indique qu'il y a eu un changement

# Logique pour mettre à jour les valeurs via un bouton (pour chaque nœud)
node_id_input = st.text_input("Entrez l'ID du nœud à modifier :", "")

# Vérifier si l'ID saisi est valide
if node_id_input:
    try:
        node_id = int(node_id_input)
        if node_id in df["marker_id"].values:
            # Afficher les colonnes de boutons pour bloquer ou forcer le nœud
            col1, col2 = st.columns(2)

            with col1:
                if st.button(f"Bloquer nœud {node_id}"):
                    update_node(
                        node_id,
                        1,
                        df.loc[df["marker_id"] == node_id, "forcé"].values[0],
                    )
                    st.session_state.data_changed = (
                        True  # Indique qu'il y a eu un changement
                    )
                    st.rerun()  # Rafraîchir automatiquement

            with col2:
                if st.button(f"Forcer nœud {node_id}"):
                    update_node(
                        node_id,
                        df.loc[df["marker_id"] == node_id, "bloqué"].values[0],
                        1,
                    )
                    st.session_state.data_changed = (
                        True  # Indique qu'il y a eu un changement
                    )
                    st.rerun()  # Rafraîchir automatiquement

            # Ajouter des boutons pour réinitialiser "bloqué" ou "forcé"
            col3, col4 = st.columns(2)

            with col3:
                if st.button(f"Réinitialiser bloqué nœud {node_id}"):
                    reset_node(node_id, reset_bloque=True, reset_force=False)
                    st.session_state.data_changed = (
                        True  # Indique qu'il y a eu un changement
                    )
                    st.rerun()  # Rafraîchir automatiquement

            with col4:
                if st.button(f"Réinitialiser forcé nœud {node_id}"):
                    reset_node(node_id, reset_bloque=False, reset_force=True)
                    st.session_state.data_changed = (
                        True  # Indique qu'il y a eu un changement
                    )
                    st.rerun()  # Rafraîchir automatiquement
        else:
            st.error("L'ID du nœud saisi n'existe pas.")
    except ValueError:
        st.error("Veuillez entrer un ID valide.")

# Créer la carte
m = folium.Map(location=[48.8566, 2.3522], zoom_start=12)

# Ajouter des marqueurs en fonction des nœuds de votre base de données
for index, row in df.iterrows():
    node_id = row["marker_id"]  # ID du nœud
    lat, lon = row["latitude"], row["longitude"]

    # Déterminer la couleur du marqueur en fonction de l'état
    if row["bloqué"]:
        color = "green"  # Couleur pour "bloqué"
    elif row["forcé"]:
        color = "orange"  # Couleur pour "forcé"
    else:
        color = "blue"  # Couleur par défaut

    # Ajouter le marqueur
    marker = folium.Marker([lat, lon], icon=folium.Icon(color=color))
    marker.add_child(
        folium.Popup(
            f"Nœud {node_id}<br>Bloqué: {row['bloqué']}<br>Forcé: {row['forcé']}"
        )
    )
    m.add_child(marker)

# Afficher la carte dans Streamlit
st_folium(m)
