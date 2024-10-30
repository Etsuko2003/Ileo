import os
import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import matplotlib.pyplot as plt
from math import floor
from io import BytesIO

from scripts.graph_coverage import (
    calculate_node_coverages,
    calculate_coverage_percentage,
    create_graph,
    filter_vanne_nodes,
    get_prioritized_nodes,
    greedy_node_selection,
)
from scripts.preprocessing import combine_dataframe, create_columns
from scripts.visualization import plot_graph_on_folium, plot_solution_on_folium
from scripts.constants import *

maps_directory = "data/maps"
if not os.path.exists(maps_directory):
    os.makedirs(maps_directory)

# Interface Streamlit
st.title("Analyse de couverture des canalisations")
pd.set_option("styler.render.max_elements", 10000000)


@st.cache_data
def load_data():
    with pd.ExcelFile(VANNE_DATA_PATH) as vanne_file, pd.ExcelFile(
        CAN_DATA_PATH
    ) as cana_file:
        vanne_df = (
            vanne_file.parse()
        )  # Ici, vous pouvez spécifier le nom de la feuille si nécessaire
        cana_df = cana_file.parse()  # Pareil ici
        # cana_df = canalisation_with_latitude_longitude(cana_df)

    return vanne_df, cana_df


vanne_df, cana_df = load_data()

# Vérifier si les données ont été chargées
if vanne_df is not None and cana_df is not None:
    st.success("Les données ont été chargées avec succès.")
    # Vous pouvez maintenant utiliser vanne_df et cana_df pour le reste de votre application
else:
    st.error("Erreur lors du chargement des données.")

vanne_df = create_columns(vanne_df)

exist = False

if os.path.exists(GESTION_DATA_PATH):
    gestion_df = pd.read_excel(GESTION_DATA_PATH)
    vanne_df = combine_dataframe(vanne_df, gestion_df)
    exist = True

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

        .button-left {
        background-color: #1E90FF; /* Bleu */
        color: white !important;
        padding: 8px 16px;
        border-radius: 8px;
        border: none;
        }
        .button-right {
        background-color: #32CD32; /* Vert */
        color: white !important;
        padding: 8px 16px;
        border-radius: 8px;
        border: none;
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
        f"data/created/{commune_name}_noeud_selected.xlsx", index=False
    )


def run_coverage_analysis(
    covering_max_distance: float, selected_communes: list[str], num_prelocators: int
):
    """Exécute l'analyse de couverture pour une commune sélectionnée."""

    # Validation des paramètres d'entrée
    if not selected_communes:
        raise ValueError("Veuillez sélectionner au moins une commune.")
    if num_prelocators <= 0:
        raise ValueError("Le nombre de prélocalisateurs doit être supérieur à zéro.")

    st.write(
        f"Traitement des communes: {', '.join(selected_communes)}"
    )  # Affiche les communes en cours de traitement
    df_commune = filter_data_by_commune(cana_df, selected_communes)

    # Validation du contenu de df_commune
    if df_commune.empty:
        st.write("Aucune donnée pour les communes sélectionnées.")
        return  # Fin de l'exécution si aucune donnée n'est présente

    # Créer le graphe pour la commune
    G = create_graph(df_commune)
    vanne_nodes = filter_vanne_nodes(G, vanne_df)

    # Calculer la couverture
    prioritized_nodes = get_prioritized_nodes(G, vanne_nodes, vanne_df)
    node_coverage, forced_nodes = calculate_node_coverages(
        G, prioritized_nodes, vanne_df, df_commune, covering_max_distance
    )

    # Appliquer la fonction de placement glouton et de calcul de couverture pour tous les prélocalisateurs
    selected_nodes, covered_ids = greedy_node_selection(
        node_coverage, num_prelocators, forced_nodes
    )

    # Sauvegarder les nœuds sélectionnés dans un fichier Excel
    save_selected_nodes_to_excel(selected_nodes, "")

    # Charger les nœuds sélectionnés depuis le fichier Excel
    try:
        st.session_state.selected_nodes_df = pd.read_excel(
            "data/created/_noeud_selected.xlsx"
        )
    except FileNotFoundError:
        st.write("Erreur : le fichier '_noeud_selected.xlsx' est introuvable.")
        return
    except ValueError as e:
        st.write(f"Erreur de lecture du fichier Excel : {e}")
        return

    # Vérifier les colonnes du DataFrame après chargement
    st.session_state.selected_nodes_df["FORCE"] = st.session_state.selected_nodes_df[
        "FORCE"
    ].astype(bool)
    st.session_state.selected_nodes_df["BLOQUE"] = st.session_state.selected_nodes_df[
        "BLOQUE"
    ].astype(bool)

    # Affichage de la solution sur une carte Folium
    st.write(
        f"Affichage de la solution sur la carte pour les communes sélectionnées..."
    )

    # Affichage de la carte
    m = plot_solution_on_folium(G, selected_nodes, covered_ids, vanne_nodes)
    map_file_path = f"data/maps/{', '.join(selected_communes)}_map.html"
    m.save(map_file_path)

    st.session_state.map_object = m


# Initialise les variables d'état de session si elles n'existent pas encore
if "show_prelocator_form" not in st.session_state:
    st.session_state.show_prelocator_form = False


# Formulaire pour la sélection des communes
with st.form(key="commune_selection_form"):
    # Sélectionnez les communes
    communes = cana_df["COMMUNE"].unique()
    selected_communes = st.multiselect(
        "Sélectionnez une ou plusieurs communes",
        options=communes,
        default=communes[0] if len(communes) > 0 else None,
    )

    col_left, col_space, col_right = st.columns([3, 7, 4])

    with col_left:
        # Bouton pour générer le plot
        submit_button_plot = st.form_submit_button(label="Générer le Plot")

    if submit_button_plot and selected_communes:
        df_commune = filter_data_by_commune(cana_df, selected_communes)

        # Calcul de la longueur totale
        total_length = calculate_total_length(df_commune)

        # Créer le graphe pour la commune
        G = create_graph(df_commune)

        vanne_nodes = filter_vanne_nodes(G, vanne_df)

        # Calculer la couverture
        prioritized_nodes = get_prioritized_nodes(G, vanne_nodes, vanne_df)
        node_coverage, forced_nodes = calculate_node_coverages(
            G, prioritized_nodes, vanne_df, df_commune, DEFAULT_COVERAGE_DISTANCE
        )

        max_prelocs = get_max_prelocators(total_length)
        max_prelocators = max_prelocs if max_prelocs > 20 else 20
        total_ids = df_commune["ID_CANA"].unique()

        # Appliquer la fonction de placement glouton et de calcul de couverture pour tous les prélocalisateurs
        num_prelocators = range(0, max_prelocators + 1)

        coverage_results_df = pd.DataFrame({"num_prelocators": num_prelocators})
        coverage_results_df["selected_nodes"], coverage_results_df["covered_ids"] = zip(
            *coverage_results_df["num_prelocators"].apply(
                lambda x: greedy_node_selection(node_coverage, x, forced_nodes)
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
    with col_right:
        # Bouton pour afficher le formulaire de génération de la carte
        submit_continue = st.form_submit_button(label="Génération de la carte")

    if submit_continue:

        df_commune = filter_data_by_commune(cana_df, selected_communes)

        # Calcul de la longueur totale
        total_length = calculate_total_length(df_commune)
        st.session_state.show_prelocator_form = (
            True  # Active l'affichage du deuxième formulaire
        )
        st.write(
            f"Vous avez sélectionné les communes `{', '.join(selected_communes)}`. Passons à la carte..."
        )

# Affichage du formulaire de sélection des prélocalisateurs si le bouton a été cliqué
if st.session_state.show_prelocator_form:
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

        # Bouton pour exécuter l'analyse de couverture et générer la carte
        submit_button_analysis = st.form_submit_button(label="Générer la map")

        if submit_button_analysis:
            try:
                covering_max_distance = (
                    float(covering_max_distance)
                    if covering_max_distance
                    else DEFAULT_COVERAGE_DISTANCE
                )
                if covering_max_distance <= 0:
                    raise ValueError(
                        "La distance maximale de couverture doit être supérieure à zéro."
                    )

                st.write(
                    "Distance maximale convertie en float : ", covering_max_distance
                )

                run_coverage_analysis(
                    covering_max_distance, selected_communes, num_prelocators
                )

                # Chemin du fichier de la carte générée
                map_file_path = f"data/maps/{', '.join(selected_communes)}_map.html"

                # Vérifie si la carte a bien été créée avant de proposer le téléchargement
                if os.path.exists(map_file_path):
                    st.session_state.map_file_path = (
                        map_file_path  # Stocke le chemin dans l'état de session
                    )
                else:
                    st.error("Le fichier de carte n'a pas été créé.")

            except ValueError as e:
                st.error(e)

    if "map_object" in st.session_state:
        st.subheader("Carte générée")
        # Afficher à nouveau l'objet de carte stocké
        st_folium(st.session_state.map_object, width=725)

    # Bouton de téléchargement en dehors du formulaire
if "map_file_path" in st.session_state:
    st.download_button(
        label="Télécharger la carte en HTML",
        data=open(st.session_state.map_file_path, "rb").read(),
        file_name=f"{', '.join(selected_communes)}_map.html",
        mime="text/html",
    )

    # Affichage de la dataframe sélectionnée en dehors du formulaire
if "selected_nodes_df" in st.session_state:
    st.subheader("Nœuds Sélectionnés")
    st.dataframe(
        st.session_state.selected_nodes_df.style.format(precision=0, thousands="")
    )

    # Section pour modifier la dataframe vanne_df
    st.subheader("Modifier les données des vannes")

    # Utiliser un data editor pour afficher et modifier vanne_df

    vanne_df_copy = (
        vanne_df.copy()
    )  # Faire une copie pour éviter de modifier l'original avant validation

    edited_vanne_df = st.data_editor(
        vanne_df_copy.style.format(precision=0, thousands=""),
        column_config={
            "ID_NOEUD": st.column_config.NumberColumn("ID_NOEUD"),
            "FORCE": st.column_config.CheckboxColumn("FORCE"),
            "BLOQUE": st.column_config.CheckboxColumn("BLOQUE"),
            # Ajoutez d'autres colonnes ici si nécessaire
        },
        hide_index=True,
        num_rows="dynamic",
    )

    # Si le bouton "Sauvegarder les modifications" est cliqué
    if st.button("Sauvegarder les modifications"):
        # Mettre à jour vanne_df avec les données éditées
        vanne_df.update(edited_vanne_df)

        # Enregistrer le DataFrame modifié dans un fichier Excel
        vanne_df.to_excel(GESTION_DATA_PATH, index=False)
        st.success("Modifications sauvegardées avec succès.")
