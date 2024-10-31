from math import floor
import pandas as pd, streamlit as st
from scripts.constants import *


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
        f"La somme des longueurs pour la commune sélectionnée est de `{total_length}` mètres."
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
    # Créer un DataFrame à partir des listes
    data = pd.DataFrame(
        {
            "Nombre de prélocalisateurs": prelocators_list,
            "Pourcentage de couverture": coverage_percentage_list,
        }
    )

    # Afficher le titre
    st.subheader(
        f"Relation entre le nombre de prélocalisateurs et la couverture de `{commune_name}`"
    )

    # Afficher le graphique avec line_chart
    st.line_chart(data.set_index("Nombre de prélocalisateurs"), color="#FF4B4B")

    # Ajouter un commentaire sur la couverture
    st.write(
        "Cette courbe montre la relation entre le nombre de prélocalisateurs et le pourcentage de couverture."
    )
    st.write(
        "Un nombre croissant de prélocalisateurs augmente généralement le pourcentage de couverture."
    )


def save_selected_nodes_to_excel(
    selected_nodes: list[str], commune_name: str, vanne_df
) -> None:
    """Sauvegarder les nœuds sélectionnés dans un fichier Excel"""
    vanne_df[vanne_df["ID_NOEUD"].isin(selected_nodes)].to_excel(
        f"data/created/{commune_name}_noeud_selected.xlsx"
    )
