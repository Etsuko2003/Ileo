import os
import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import matplotlib.pyplot as plt
from math import floor

from scripts.graph_coverage import (
    calculate_node_coverages,
    calculate_coverage_percentage,
    create_graph,
    get_prioritized_nodes,
    greedy_node_selection,
)
from scripts.preprocessing import canalisation_with_latitude_longitude
from scripts.visualization import plot_graph_on_folium, plot_solution_on_folium
from scripts.constants import *

maps_directory = "data/maps"
if not os.path.exists(maps_directory):
    os.makedirs(maps_directory)

# Interface Streamlit
st.title("Analyse de couverture des canalisations")


import os
import streamlit as st
import pandas as pd


@st.cache_data
def load_data():
    # Affiche le chemin des fichiers
    st.markdown(f"**Chargement des fichiers depuis :**")
    st.markdown(f"- Vanne: `{VANNE_DATA_PATH}`")
    st.markdown(f"- Canalisation: `{CAN_DATA_PATH}`")
    st.markdown(f"- Noeud: `{NOEUD_DATA_PATH}`")

    with pd.ExcelFile(VANNE_DATA_PATH) as vanne_file, pd.ExcelFile(
        CAN_DATA_PATH
    ) as cana_file:
        vanne_df = (
            vanne_file.parse()
        )  # Ici, vous pouvez spécifier le nom de la feuille si nécessaire
        cana_df = cana_file.parse()  # Pareil ici
        # cana_df = canalisation_with_latitude_longitude(cana_df)

    return vanne_df, cana_df


def check_and_load_data():
    # Vérifiez si les fichiers existent
    files_exist = os.path.exists(VANNE_DATA_PATH) and os.path.exists(CAN_DATA_PATH)

    if files_exist:
        # Demander à l'utilisateur s'il veut recharger les fichiers
        reload_data = st.radio(
            "Les fichiers existent déjà. Voulez-vous les recharger ?", ("Oui", "Non")
        )

        if reload_data == "Oui":
            # Si l'utilisateur choisit de recharger, alors charger les données
            return load_data()
        else:
            # Si l'utilisateur ne veut pas recharger, charger les données existantes
            st.write("Chargement des fichiers existants...")
            vanne_df = pd.read_excel(VANNE_DATA_PATH)
            cana_df = pd.read_excel(CAN_DATA_PATH)
            return vanne_df, cana_df
    else:
        # Si les fichiers n'existent pas, informer l'utilisateur
        st.warning(
            "Les fichiers nécessaires n'existent pas. Veuillez les télécharger et les placer dans le dossier approprié."
        )
        return None, None


def upload_files():
    """Fonction pour uploader les fichiers vanne et canalisation."""
    st.subheader("Télécharger de nouveaux fichiers")
    vanne_file = st.file_uploader("Téléchargez le fichier Vanne", type=["xlsx"])
    cana_file = st.file_uploader("Téléchargez le fichier Canalisation", type=["xlsx"])

    if vanne_file and cana_file:
        # Sauvegarder les fichiers dans le dossier data
        with open(VANNE_DATA_PATH, "wb") as f:
            f.write(vanne_file.getbuffer())
        with open(CAN_DATA_PATH, "wb") as f:
            f.write(cana_file.getbuffer())
        st.success("Les fichiers ont été téléchargés avec succès.")


# Vérification et chargement des données
if st.button("Charger les données existantes ou télécharger de nouveaux fichiers"):
    upload_files()

vanne_df, cana_df = check_and_load_data()

# Vérifier si les données ont été chargées
if vanne_df is not None and cana_df is not None:
    st.success("Les données ont été chargées avec succès.")
    # Vous pouvez maintenant utiliser vanne_df et cana_df pour le reste de votre application
else:
    st.error("Erreur lors du chargement des données.")


# CSS to widen the main container and DataFrame
st.markdown(
    """
    <style>
        /* Expanding main content area */
        .main .block-container {
            max-width: 95%;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        /* Expanding form container */
        .stForm {
            max-width: 100% !important;
            width: 100% !important;
        }

        /* Widening the form fields */
        .stForm .stTextInput, .stForm .stSelectbox, .stForm .stButton, .stForm .stNumberInput {
            width: 100% !important;
            max-width: 100% !important;
        }

        /* Adjusting buttons to full width */
        .stForm .stButton>button {
            width: 100% !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def display_commune_options(communes: list[str]) -> str:
    """Afficher les options de communes avec un selectbox"""
    return st.selectbox("Sélectionnez la commune :", ["Toute la région"] + communes)


def filter_data_by_commune(
    df: pd.DataFrame, selected_communes: list[str]
) -> pd.DataFrame:
    """Filtrer les données en fonction de la commune choisie"""
    return (
        df
        if selected_communes == ["Toute la région"]
        else df[df["COMMUNE"].isin(selected_communes)]
    )


def calculate_total_length(df_commune: pd.DataFrame) -> float:
    """Calculer la somme des longueurs de canalisation pour une commune"""
    total_length = df_commune["LONGUEUR_EN_M"].sum()
    st.write(
        f"La somme des longueurs pour la commune sélectionnée est de {total_length} mètres."
    )
    return total_length


def get_max_prelocators(total_length: float) -> int:
    """Calculer le nombre maximum de prélocalisateurs en fonction de la longueur totale"""
    return floor(total_length / MAX_LENGTH_THRESHOLD * MIN_PRELOCATORS)


def generate_coverage_plot(
    prelocators_list: list[int],
    coverage_percentage_list: list[float],
    max_prelocators: int,
    commune_name: str,
) -> None:
    """Générer et afficher une courbe de couverture avec style personnalisé"""
    # Définir les dimensions et le style de la figure
    plt.figure(figsize=(14, 8))  # Taille de la figure pour un affichage plus grand
    plt.style.use(
        "seaborn-v0_8-darkgrid"
    )  # Appliquer un style pour un rendu plus professionnel

    # Tracer la courbe avec des options de style
    plt.plot(
        prelocators_list,
        coverage_percentage_list,
        marker="o",
        markersize=6,
        linewidth=2,
        color=(1.0, 0.294, 0.294),
        label="Pourcentage de couverture",
    )

    # Titre et labels avec tailles de police personnalisées
    plt.title(
        f"Relation entre le nombre de prélocalisateurs et la couverture - {commune_name}",
        fontsize=18,
        fontweight="bold",
        color="navy",
    )
    plt.xlabel("Nombre de prélocalisateurs", fontsize=14, labelpad=10)
    plt.ylabel("Pourcentage de couverture (%)", fontsize=14, labelpad=10)

    # Limites et échelles
    plt.xlim(0, max_prelocators)
    plt.ylim(0, 100)
    plt.xticks(range(0, max_prelocators + 1, 5), fontsize=12)
    plt.yticks(range(0, 101, 10), fontsize=12)

    # Grille plus prononcée
    plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    # Ajouter des annotations pour les points importants
    for i, (x, y) in enumerate(zip(prelocators_list, coverage_percentage_list)):
        plt.annotate(
            "",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
            color="darkblue",
        )
    plt.legend(fontsize=12)
    plt.tight_layout()
    st.pyplot(plt, use_container_width=True)


def save_selected_nodes_to_excel(selected_nodes: list[str], commune_name: str) -> None:
    """Sauvegarder les nœuds sélectionnés dans un fichier Excel"""
    vanne_df[vanne_df["ID_NOEUD"].isin(selected_nodes)].to_excel(
        f"data/created/{commune_name}_noeud_selected.xlsx"
    )


def run_coverage_analysis(
    covering_max_distance: float, selected_communes: list[str], num_prelocators: int
):
    """Exécute l'analyse de couverture pour une commune sélectionnée."""
    st.write(
        f"Traitement des communes: {', '.join(selected_communes)}"
    )  # Affiche les communes en cours de traitement
    df_commune = filter_data_by_commune(cana_df, selected_communes)

    # Calcul de la longueur totale
    total_length = calculate_total_length(df_commune)

    # Créer le graphe pour la commune
    G = create_graph(df_commune)

    combined_nodes = pd.concat([df_commune["ID_NOEUD_1"], df_commune["ID_NOEUD_2"]])
    vanne_in_cana = vanne_df[vanne_df["ID_NOEUD"].isin(combined_nodes)]
    vanne_nodes = set(vanne_in_cana["ID_NOEUD"])

    # Calculer la couverture
    prioritized_nodes = get_prioritized_nodes(G, vanne_nodes)
    node_coverage = calculate_node_coverages(
        G, prioritized_nodes, df_commune, covering_max_distance
    )

    max_prelocators = get_max_prelocators(total_length)
    total_ids = df_commune["ID_CANA"].unique()

    # Appliquer la fonction de placement glouton et de calcul de couverture pour tous les prélocalisateurs
    selected_nodes, covered_ids = greedy_node_selection(node_coverage, num_prelocators)

    # Sauvegarder les nœuds sélectionnés dans un fichier Excel
    save_selected_nodes_to_excel(selected_nodes, ", ".join(selected_communes))

    # Afficher la solution sur une carte Folium
    st.write(
        f"Affichage de la solution sur la carte pour les communes sélectionnées..."
    )

    with st.container():
        # Affichage de la carte
        m = plot_solution_on_folium(G, selected_nodes, covered_ids, vanne_nodes)
        map_file_path = f"data/maps/{', '.join(selected_communes)}_map.html"
        m.save(map_file_path)
        st_folium(m, width=725)

        # Affichage du tableau
        st.subheader("Nœuds Sélectionnés")
        selected_nodes_df = vanne_df[vanne_df["ID_NOEUD"].isin(selected_nodes)]
        st.dataframe(selected_nodes_df)


# Formulaire pour la sélection des communes
with st.form(key="commune_selection_form"):
    # Sélectionnez les communes
    communes = cana_df["COMMUNE"].unique()
    selected_communes = st.multiselect(
        "Sélectionnez une ou plusieurs communes",
        options=communes,
        default=communes[0] if len(communes) > 0 else None,
    )

    # Bouton de validation du formulaire pour générer le plot
    submit_button_plot = st.form_submit_button(label="Générer le Plot")

    if submit_button_plot and selected_communes:
        df_commune = filter_data_by_commune(
            cana_df, selected_communes
        )  # Get the first selected commune for initial calculations

        # Calcul de la longueur totale
        total_length = calculate_total_length(df_commune)

        # Créer le graphe pour la commune
        G = create_graph(df_commune)

        combined_nodes = pd.concat([df_commune["ID_NOEUD_1"], df_commune["ID_NOEUD_2"]])
        vanne_in_cana = vanne_df[vanne_df["ID_NOEUD"].isin(combined_nodes)]
        vanne_nodes = set(vanne_in_cana["ID_NOEUD"])

        # Calculer la couverture
        prioritized_nodes = get_prioritized_nodes(G, vanne_nodes)
        node_coverage = calculate_node_coverages(
            G, prioritized_nodes, df_commune, DEFAULT_COVERAGE_DISTANCE
        )

        max_prelocs = get_max_prelocators(total_length)
        max_prelocators = max_prelocs if max_prelocs > 20 else 20
        total_ids = df_commune["ID_CANA"].unique()

        # Appliquer la fonction de placement glouton et de calcul de couverture pour tous les prélocalisateurs
        num_prelocators = range(0, max_prelocators + 1)

        coverage_results_df = pd.DataFrame({"num_prelocators": num_prelocators})
        coverage_results_df["selected_nodes"], coverage_results_df["covered_ids"] = zip(
            *coverage_results_df["num_prelocators"].apply(
                lambda x: greedy_node_selection(node_coverage, x)
            )
        )
        coverage_results_df["coverage_percentage"] = coverage_results_df[
            "covered_ids"
        ].apply(
            lambda covered_ids: calculate_coverage_percentage(total_ids, covered_ids)
        )

        # Tracer la courbe de couverture
        generate_coverage_plot(
            coverage_results_df["num_prelocators"].tolist(),
            coverage_results_df["coverage_percentage"].tolist(),
            max_prelocators,
            ", ".join(selected_communes),
        )

# Formulaire pour la sélection du nombre de prélocalisateurs
with st.form(key="prelocator_selection_form"):
    num_prelocators = st.number_input(
        "Combien de prélocalisateurs souhaitez-vous déployer ?",
        min_value=1,
        max_value=200,
        value=1,
    )

    covering_max_distance = st.text_input(
        "Saisissez la distance maximale que doivent couvrir les prélocalisateurs (ou laissez vide pour utiliser la valeur par défaut)",
        value=str(DEFAULT_COVERAGE_DISTANCE),
    )

    # Bouton de validation du formulaire pour exécuter l'analyse
    submit_button_analysis = st.form_submit_button(label="Exécuter l'Analyse")

    if submit_button_analysis and selected_communes:
        try:
            covering_max_distance = float(covering_max_distance)
            st.write("Distance maximale convertie en float : ", covering_max_distance)
            run_coverage_analysis(
                covering_max_distance, selected_communes, num_prelocators
            )
            # Chemin du fichier de la carte
            map_file_path = f"data/maps/{', '.join(selected_communes)}_map.html"

            # Vérifiez si la carte a bien été créée avant de proposer le téléchargement
            if os.path.exists(map_file_path):
                st.session_state.map_file_path = (
                    map_file_path  # Stockez le chemin dans l'état de session
                )
            else:
                st.error("Le fichier de carte n'a pas été créé.")

        except ValueError:
            st.error("Veuillez saisir une distance valide.")

# Bouton de téléchargement (en dehors du formulaire)
if "map_file_path" in st.session_state:
    # Bouton de téléchargement
    st.download_button(
        label="Télécharger la carte en HTML",
        data=open(st.session_state.map_file_path, "rb").read(),
        file_name=f"{', '.join(selected_communes)}_map.html",
        mime="text/html",
    )
