import os
import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import matplotlib.pyplot as plt
from math import floor

pd.set_option("styler.render.max_elements", 330808)

from scripts.graph_coverage import (
    calculate_node_coverages,
    calculate_coverage_percentage,
    create_graph,
    filter_vanne_nodes,
    get_prioritized_nodes,
    greedy_node_selection,
)
from scripts.preprocessing import (
    canalisation_with_latitude_longitude,
    combine_dataframe,
    create_columns,
)
from scripts.visualization import plot_solution_on_folium
from scripts.coverage_analysis import *
from scripts.constants import *

maps_directory = "data/maps"
if not os.path.exists(maps_directory):
    os.makedirs(maps_directory)

# Interface Streamlit
st.title("Analyse de couverture des canalisations")


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

if os.path.exists(GESTION_DATA_PATH):
    gestion_df = pd.read_excel(GESTION_DATA_PATH)
    vanne_df = combine_dataframe(vanne_df, gestion_df)

# Define the editor for vanne_df
st.header("Éditeur de données Vanne")
vanne_df[["ID_NOEUD", "ID_VANNE"]] = vanne_df[["ID_NOEUD", "ID_VANNE"]].astype(int)
edited_vanne_df = st.data_editor(
    vanne_df[["ID_NOEUD", "ID_VANNE", "BLOQUE", "FORCE", "LOCALISATION"]],
    num_rows="fixed",
    use_container_width=True,
    hide_index=True,
    disabled=["ID_NOEUD", "ID_VANNE"],
    key="vanne_data_editor",
)
col_left, col_space, col_right = st.columns([5, 2, 4])
with col_left:
    # Button to update the vanne_df and rerun the analysis
    if st.button("Mettre à jour la carte"):
        # Convert edited_vanne_df to a DataFrame
        edited_vanne_df = pd.DataFrame(edited_vanne_df)

        # Merge with the original DataFrame to keep all columns
        updated_vanne_df = vanne_df.merge(
            edited_vanne_df,
            on=["ID_NOEUD", "ID_VANNE"],
            how="left",
            suffixes=("", "_edited"),
        )

        # Select the original columns and update the edited ones
        for col in ["BLOQUE", "FORCE", "LOCALISATION"]:
            updated_vanne_df[col] = updated_vanne_df[f"{col}_edited"].combine_first(
                updated_vanne_df[col]
            )

        # Drop the temporary columns created by the merge
        updated_vanne_df = updated_vanne_df[vanne_df.columns]

        # Store the updated DataFrame back to the session state
        st.session_state.vanne_df = updated_vanne_df
        st.success("Les données de vanne ont été mises à jour avec succès.")

        # Rerun the app to reflect changes
        st.rerun()

# Check if the session state has an updated vanne_df
if "vanne_df" in st.session_state:
    vanne_df = st.session_state.vanne_df
with col_space:
    st.write("       ")
with col_right:
    if st.button("Sauvegarder les modifications"):
        # Mettre à jour vanne_df avec les données éditées
        vanne_df.update(edited_vanne_df)

        # Enregistrer le DataFrame modifié dans un fichier Excel
        vanne_df.to_excel(GESTION_DATA_PATH, index=False)
        st.write("Vous retrouverez votre fichier ici :`data/GESTION.xlsx`.")


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
    save_selected_nodes_to_excel(selected_nodes, "", vanne_df)

    # Afficher la solution sur une carte Folium
    st.write(
        f"Affichage de la solution sur la carte pour les communes sélectionnées..."
    )

    # Affichage de la carte
    m = plot_solution_on_folium(G, selected_nodes, covered_ids, vanne_nodes)
    map_file_path = f"data/maps/{', '.join(selected_communes)}_map.html"
    m.save(map_file_path)
    st_folium(m, width=725)

    # Affichage du tableau
    st.session_state.selected_nodes_df = vanne_df[
        vanne_df["ID_NOEUD"].isin(selected_nodes)
    ]


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
    submit_button_analysis = st.form_submit_button(label="Générer la map")

    if submit_button_analysis and selected_communes:
        try:
            covering_max_distance = (
                float(covering_max_distance)
                if covering_max_distance
                else DEFAULT_COVERAGE_DISTANCE
            )

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

if "selected_nodes_df" in st.session_state:
    st.header("Nœuds Sélectionnés")
    st.dataframe(
        st.session_state.selected_nodes_df.style.format(precision=0, thousands=""),
        hide_index=True,
    )
