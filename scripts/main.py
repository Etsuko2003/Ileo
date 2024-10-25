from math import floor
import matplotlib.pyplot as plt
import pandas as pd

from graph_coverage import (
    calculate_node_coverages,
    calculate_coverage_percentage,
    create_graph,
    get_prioritized_nodes,
    greedy_node_selection,
)
from visualization import plot_solution_on_folium
from constants import *

# Charger les fichiers utiles
with pd.ExcelFile(VANNE_DATA_PATH) as vanne_file, pd.ExcelFile(
    CAN_DATA_PATH
) as cana_file:
    vanne_df = vanne_file.parse()
    cana_df = cana_file.parse()


def display_commune_options(communes: list[str]) -> None:
    """Affiche les options de communes avec un numéro"""
    print("0. Toute la région")
    for idx, commune in enumerate(communes, start=1):
        print(f"{idx}. {commune}")


def get_user_commune_selection(communes: list[str]) -> list[str] | str:
    """Obtenir la sélection des communes de l'utilisateur"""
    while True:
        choix = input(PROMPT_COMMUNE_SELECTION)

        try:
            # Convertir l'entrée en une liste de numéros
            choix_list = [int(num.strip()) for num in choix.split(",")]

            # Vérifier si tous les choix sont valides
            if all(1 <= choix <= len(communes) for choix in choix_list):
                return list(set(communes[choix - 1] for choix in choix_list))
            elif choix_list == [0]:
                return "All"  # Prendre toute la région en compte
            else:
                print("Erreur : Un ou plusieurs numéros en dehors des options valides.")
        except ValueError:
            print("Erreur : Entrée non valide, veuillez entrer des numéros valides.")


def filter_data_by_commune(
    df: pd.DataFrame, selected_communes: list[str] | str
) -> pd.DataFrame:
    """Filtrer les données en fonction de la commune choisie"""
    return (
        df[df["COMMUNE"].isin(selected_communes)] if selected_communes != "All" else df
    )


def calculate_total_length(df_commune: pd.DataFrame) -> float:
    """Calculer la somme des longueurs de canalisation pour une commune"""
    total_length = df_commune["LONGUEUR_EN_M"].sum()
    print(
        f"La somme des longueurs pour la ou (les) commune.s est de {total_length} mètres."
    )
    return total_length


def determine_number_of_prelocators(total_length: float) -> int:
    """Demander à l'utilisateur le nombre de prélocalisateurs à déployer"""
    while True:
        prompt = (
            PROMPT_NUM_PRELOCATORS_UNDER_THRESHOLD
            if total_length < MAX_LENGTH_THRESHOLD
            else PROMPT_NUM_PRELOCATORS_OVER_THRESHOLD
        )
        num_prelocators = input(prompt)

        try:
            num_prelocators = int(num_prelocators)
            print(f"Vous allez déployer {num_prelocators} prélocalisateurs.")
            return num_prelocators
        except ValueError:
            print("Erreur : Entrez un nombre valide.")


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
    plt.show()


def save_selected_nodes_to_excel(selected_nodes: list[str], commune_name: str) -> None:
    """Sauvegarder les nœuds sélectionnés dans un fichier Excel"""
    vanne_df[vanne_df["ID_NOEUD"].isin(selected_nodes)].to_excel(
        f"data/created/{commune_name}_noeud_selected.xlsx"
    )


def run_coverage_analysis_once() -> None:
    """Exécute l'analyse de couverture pour une seule commune."""
    covering_max_distance = (
        input(
            f"Saisissez la distance maximale que doivent couvrir les prélocalisateurs ou la touche 'Enter' si {DEFAULT_COVERAGE_DISTANCE} vous va : "
        )
        or DEFAULT_COVERAGE_DISTANCE
    )
    covering_max_distance = float(covering_max_distance)

    df = cana_df

    # Lister les communes sans doublon
    communes = df["COMMUNE"].unique()

    # Afficher les options
    display_commune_options(communes)

    # Obtenir la commune sélectionnée
    selected_commune = get_user_commune_selection(communes)
    df_commune = filter_data_by_commune(df, selected_commune)

    # Calculer la longueur totale
    total_length = calculate_total_length(df_commune)

    # Créer le graphe pour la commune
    G = create_graph(df_commune)
    
    combined_nodes = pd.concat([df_commune["ID_NOEUD_1"], df_commune["ID_NOEUD_2"]])
    vanne_in_cana = vanne_df[vanne_df["ID_NOEUD"].isin(combined_nodes)]
    vanne_nodes = set(vanne_in_cana["ID_NOEUD"])

    # Calculer la couverture
    prioritized_nodes = get_prioritized_nodes(G, vanne_nodes)
    node_coverage = calculate_node_coverages(
        G, prioritized_nodes, df, covering_max_distance
    )

    max_prelocators = get_max_prelocators(total_length)
    total_ids = df_commune["ID_CANA"].unique()

    # Appliquer la fonction de placement glouton et de calcul de couverture pour tous les prélocalisateurs
    coverage_results = pd.DataFrame({"num_prelocators": range(0, max_prelocators + 1)})

    coverage_results["selected_nodes"], coverage_results["covered_ids"] = zip(
        *coverage_results["num_prelocators"].apply(
            lambda x: greedy_node_selection(node_coverage, x)
        )
    )

    coverage_results["coverage_percentage"] = coverage_results["covered_ids"].apply(
        lambda covered_ids: calculate_coverage_percentage(total_ids, covered_ids)
    )

    # Afficher les résultats
    coverage_results.apply(
        lambda row: print(
            f"Pourcentage de couverture des canalisations: {row['coverage_percentage']:.2f}% pour {len(row['selected_nodes'])} prélocalisateurs"
        ),
        axis=1,
    )

    # Tracer la courbe de couverture
    generate_coverage_plot(
        coverage_results["num_prelocators"].tolist(),
        coverage_results["coverage_percentage"].tolist(),
        max_prelocators,
        selected_commune,
    )

    # Demander le nombre de prélocalisateurs à déployer
    num_prelocators = determine_number_of_prelocators(total_length)

    # Sélectionner les meilleurs nœuds pour le nombre choisi de prélocalisateurs
    selected_nodes, covered_ids = greedy_node_selection(node_coverage, num_prelocators)

    # Calculer et afficher le pourcentage de couverture
    percentage_covered = calculate_coverage_percentage(total_ids, covered_ids)
    print(
        f"Pourcentage de couverture des canalisations: {percentage_covered:.2f}% pour {len(selected_nodes)} prélocalisateurs"
    )

    # Sauvegarder les nœuds sélectionnés dans un fichier Excel
    save_selected_nodes_to_excel(selected_nodes, selected_commune)

    # Afficher la solution sur une carte Folium et sauvegarder en HTML
    m = plot_solution_on_folium(G, selected_nodes, covered_ids, vanne_nodes)
    m.save(f"maps/solution_couverture_comparaison_{selected_commune}.html")
    print("Fichier 'solution_couverture_comparaison.html' créé.")


def run_coverage_analysis_with_retry() -> None:
    """Gestion du recommencement pour l'analyse de couverture."""
    while True:
        run_coverage_analysis_once()

        # Option pour recommencer ou arrêter
        continuer = input("Voulez-vous recommencer ? (o/n) : ").lower()
        if continuer != "o":
            print("Programme terminé.")
            break


if __name__ == "__main__":
    run_coverage_analysis_with_retry()
