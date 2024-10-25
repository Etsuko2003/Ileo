import pandas as pd
import networkx as nx
from scripts.constants import MAX_DIAMETER_TO_CONSIDER


# 1. Création du graphe à partir du DataFrame
def create_graph(df: pd.DataFrame) -> nx.Graph:
    """Créer un graphe à partir des données des canalisations."""
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row["ID_NOEUD_1"], pos=(row["LATITUDE_1"], row["LONGITUDE_1"]))
        G.add_node(row["ID_NOEUD_2"], pos=(row["LATITUDE_2"], row["LONGITUDE_2"]))
        G.add_edge(
            row["ID_NOEUD_1"],
            row["ID_NOEUD_2"],
            length=row["LONGUEUR_EN_M"],
            ID_CANA=row["ID_CANA"],
            diameter=row["DIAMETRE"],
        )
    return G


# 2. Filtrer les nœuds ayant une vanne
def filter_vanne_nodes(G: nx.Graph, df_vanne: pd.DataFrame) -> set:
    """Récupérer les nœuds dans le graphe où des vannes sont présentes."""
    combined_nodes = set(G.nodes())
    return set(df_vanne[df_vanne["ID_NOEUD"].isin(combined_nodes)]["ID_NOEUD"])


# 3. Obtenir les nœuds prioritaires selon le critère de diamètre des canalisations
def get_prioritized_nodes(G: nx.Graph, vanne_nodes: set) -> set:
    """Obtenir les nœuds prioritaires qui sont connectés à des canalisations de diamètre <= 350 mm."""
    prioritized_nodes = {
        node
        for node in vanne_nodes
        if any(
            data["diameter"] <= MAX_DIAMETER_TO_CONSIDER
            for _, _, data in G.edges(node, data=True)
        )
    }
    return prioritized_nodes


# 4. Calculer les canalisations couvertes à partir d'un nœud donné
def compute_covered_pipelines(
    G: nx.Graph, node: str, edge_to_id: dict, max_distance: float
) -> set:
    """Calcule les canalisations couvertes à partir d'un nœud avec un critère de distance et de diamètre."""
    covered_ids = set()
    distances = nx.single_source_dijkstra_path_length(
        G, node, cutoff=max_distance, weight="length"
    )

    for neighbor in distances:
        for u, v in G.edges(neighbor):
            if (u, v) in edge_to_id:
                canal_id = edge_to_id[(u, v)]
                canal_diameter = G[u][v]["diameter"]
                if canal_diameter <= MAX_DIAMETER_TO_CONSIDER:
                    covered_ids.add(canal_id)

    return covered_ids


# 5. Calculer la couverture pour chaque nœud prioritaire
def calculate_node_coverages(
    G: nx.Graph, prioritized_nodes: set, df: pd.DataFrame, max_distance: float
) -> dict:
    """Calcule les canalisations couvertes pour chaque nœud prioritaire."""
    edge_to_id = {
        (row["ID_NOEUD_1"], row["ID_NOEUD_2"]): row["ID_CANA"]
        for _, row in df.iterrows()
    }

    return {
        node: compute_covered_pipelines(G, node, edge_to_id, max_distance)
        for node in prioritized_nodes
    }


# 6. Placement glouton des prélocalisateurs
def greedy_node_selection(node_coverage: dict, n: int) -> tuple[set, set]:
    """Sélectionne les N meilleurs nœuds pour maximiser la couverture."""
    selected_nodes = set()
    covered_ids = set()

    for _ in range(n):
        best_node = max(
            node_coverage,
            key=lambda node: len(node_coverage[node] - covered_ids),
        )
        selected_nodes.add(best_node)
        covered_ids.update(node_coverage[best_node])

    return selected_nodes, covered_ids


# 7. Calculer le pourcentage de couverture
def calculate_coverage_percentage(total_ids: set, covered_ids: set) -> float:
    """Calcule le pourcentage de canalisations couvertes."""
    return (len(covered_ids) / len(total_ids)) * 100
