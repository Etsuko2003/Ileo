import streamlit as st
from streamlit_folium import folium_static
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
from scripts.visualization import plot_solution_on_folium
from scripts.constants import *


# Charger les fichiers utiles
@st.cache_data
def load_data():
    with pd.ExcelFile(VANNE_DATA_PATH) as vanne_file, pd.ExcelFile(
        CAN_DATA_PATH
    ) as cana_file:
        vanne_df = vanne_file.parse()
        cana_df = cana_file.parse()
    return vanne_df, cana_df


vanne_df, cana_df = load_data()

# Interface Streamlit
st.title("Analyse de couverture des canalisations")


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
    """Générer et afficher une courbe de couverture"""
    plt.figure(figsize=(10, 6))
    plt.plot(prelocators_list, coverage_percentage_list, marker="o")
    plt.title(
        f"Relation entre le nombre de prélocalisateurs et la couverture - {commune_name}"
    )
    plt.xlabel("Nombre de prélocalisateurs")
    plt.ylabel("Pourcentage de couverture (%)")
    plt.grid(True)
    plt.xticks(range(0, max_prelocators + 1, 5))
    plt.yticks(range(0, 101, 10))
    plt.xlim(left=0, right=max_prelocators)
    plt.ylim(bottom=0, top=100)
    st.pyplot(plt)


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
    coverage_results_df = pd.DataFrame(
        {"num_prelocators": range(0, max_prelocators + 1)}
    )

    coverage_results_df["selected_nodes"], coverage_results_df["covered_ids"] = zip(
        *coverage_results_df["num_prelocators"].apply(
            lambda x: greedy_node_selection(node_coverage, x)
        )
    )

    coverage_results_df["coverage_percentage"] = coverage_results_df[
        "covered_ids"
    ].apply(lambda covered_ids: calculate_coverage_percentage(total_ids, covered_ids))

    # Sauvegarder les nœuds sélectionnés dans un fichier Excel
    selected_nodes, covered_ids = greedy_node_selection(node_coverage, num_prelocators)
    save_selected_nodes_to_excel(selected_nodes, ", ".join(selected_communes))

    # Afficher la solution sur une carte Folium
    st.write(
        f"Affichage de la solution sur la carte pour les communes sélectionnées..."
    )
    m = plot_solution_on_folium(G, selected_nodes, covered_ids, vanne_nodes)
    folium_static(m)

    st.success(
        "Analyse de couverture terminée."
    )  # Indication que le processus est terminé


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

        max_prelocators = get_max_prelocators(total_length)
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
        except ValueError:
            st.error("Veuillez saisir une distance valide.")
