import pandas as pd
import networkx as nx
from scripts.constants import MAX_DIAMETER_TO_CONSIDER


# 1. Création du graphe à partir du DataFrame
def create_graph(df: pd.DataFrame) -> nx.Graph:
    """Créer un graphe à partir des données des canalisations."""

    # Check for required columns and missing values
    required_columns = [
        "ID_NOEUD_1",
        "ID_NOEUD_2",
        "LATITUDE_1",
        "LONGITUDE_1",
        "LATITUDE_2",
        "LONGITUDE_2",
        "LONGUEUR_EN_M",
        "ID_CANA",
        "DIAMETRE",
    ]

    if not df[required_columns].notnull().all().all():
        missing_columns = [col for col in required_columns if df[col].isnull().any()]
        raise ValueError(
            f"DataFrame contains missing values in columns: {missing_columns}."
        )

    G = nx.Graph()
    for _, row in df.iterrows():
        if not G.has_node(row["ID_NOEUD_1"]):
            G.add_node(row["ID_NOEUD_1"], pos=(row["LATITUDE_1"], row["LONGITUDE_1"]))
        if not G.has_node(row["ID_NOEUD_2"]):
            G.add_node(row["ID_NOEUD_2"], pos=(row["LATITUDE_2"], row["LONGITUDE_2"]))

        if G.has_edge(row["ID_NOEUD_1"], row["ID_NOEUD_2"]):
            # Handle duplicate edges if needed (e.g., aggregate attributes)
            continue  # Or merge attributes based on your requirement

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

    # Vérification des types d'entrée
    if not isinstance(G, nx.Graph):
        raise TypeError("G doit être un graphe NetworkX (nx.Graph).")
    if not isinstance(df_vanne, pd.DataFrame):
        raise TypeError("df_vanne doit être un DataFrame pandas.")

    # Vérification de la présence de la colonne 'ID_NOEUD' dans df_vanne
    if "ID_NOEUD" not in df_vanne.columns:
        raise ValueError("df_vanne doit contenir une colonne 'ID_NOEUD'.")

    # Vérifier si df_vanne est vide
    if df_vanne.empty:
        return set()  # Aucun nœud avec vanne trouvé

    # Vérifier si le graphe contient des nœuds
    combined_nodes = set(G.nodes())
    if not combined_nodes:
        return set()  # Aucun nœud dans le graphe

    # Filtrer les nœuds de df_vanne présents dans le graphe
    vanne_nodes = set(df_vanne[df_vanne["ID_NOEUD"].isin(combined_nodes)]["ID_NOEUD"])

    return vanne_nodes.copy()  # Retourner une copie pour éviter les effets de bord


# 3. Obtenir les nœuds prioritaires selon le critère de diamètre des canalisations
def get_prioritized_nodes(G: nx.Graph, vanne_nodes: set, df: pd.DataFrame) -> set:
    """
    Obtenir les nœuds prioritaires qui sont connectés à des canalisations de diamètre <= 350 mm,
    en tenant compte des nœuds BLOQUE et FORCE.
    """
    prioritized_nodes = set()

    # Agréger BLOQUE et FORCE par nœud en prenant le maximum (True si présent dans au moins une ligne)
    df_aggregated = df.groupby("ID_NOEUD")[["BLOQUE", "FORCE"]].max()

    # Créer un dictionnaire de nœuds avec leurs statuts BLOQUE et FORCE
    node_status = df_aggregated.to_dict("index")

    for node in vanne_nodes:
        # Vérifier si le nœud est forcé ou bloqué
        is_forced = node_status.get(node, {}).get("FORCE", False)
        is_blocked = node_status.get(node, {}).get("BLOQUE", False)

        if is_blocked:
            continue  # Ne pas inclure les nœuds BLOQUE

        # Ajouter les nœuds forcés ou les nœuds avec le bon diamètre
        if is_forced or any(
            data["diameter"] <= MAX_DIAMETER_TO_CONSIDER
            for _, _, data in G.edges(node, data=True)
        ):
            prioritized_nodes.add(node)

    return prioritized_nodes


# 4. Calculer les canalisations couvertes à partir d'un nœud donné
def compute_covered_pipelines(
    G: nx.Graph, node: str, edge_to_id: dict, max_distance: float
) -> set:
    """Calcule les canalisations couvertes à partir d'un nœud avec un critère de distance et de diamètre."""

    # Vérifications des paramètres d'entrée
    if not isinstance(G, nx.Graph):
        raise TypeError("G doit être un graphe NetworkX.")
    if node not in G:
        raise ValueError(f"Le nœud '{node}' n'existe pas dans le graphe.")
    if not isinstance(edge_to_id, dict):
        raise TypeError("edge_to_id doit être un dictionnaire.")
    if not isinstance(max_distance, (int, float)) or max_distance <= 0:
        raise ValueError("max_distance doit être un nombre positif.")
    if (
        not isinstance(MAX_DIAMETER_TO_CONSIDER, (int, float))
        or MAX_DIAMETER_TO_CONSIDER <= 0
    ):
        raise ValueError("MAX_DIAMETER_TO_CONSIDER doit être un nombre positif.")

    covered_ids = set()

    # Calcul des distances minimales à partir du nœud donné
    distances = nx.single_source_dijkstra_path_length(
        G, node, cutoff=max_distance, weight="length"
    )

    for neighbor, distance in distances.items():
        # Parcourir chaque arête à partir du voisin à une distance admissible
        for u, v in G.edges(neighbor):
            # Vérification de la clé d'arête dans les deux directions pour les graphes non orientés
            if (u, v) not in edge_to_id and (v, u) not in edge_to_id:
                continue  # Ignore l'arête si elle n'est pas dans le dictionnaire d'identifiants

            # Récupération de l'identifiant de canalisation
            canal_id = edge_to_id.get((u, v), edge_to_id.get((v, u)))

            # Vérification de l'attribut 'diameter' pour éviter les erreurs de clé manquante
            if "diameter" not in G[u][v]:
                raise KeyError(f"L'arête {(u, v)} manque l'attribut 'diameter'.")
            canal_diameter = G[u][v]["diameter"]

            # Vérification de l'attribut 'length' pour éviter les erreurs de clé manquante
            if "length" not in G[u][v]:
                raise KeyError(f"L'arête {(u, v)} manque l'attribut 'length'.")
            canal_length = G[u][v]["length"]

            # Vérifiez que la longueur cumulée du chemin n'excède pas max_distance
            total_distance = distance + canal_length
            if (
                total_distance <= max_distance
                and canal_diameter <= MAX_DIAMETER_TO_CONSIDER
            ):
                covered_ids.add(canal_id)

    return covered_ids


# 5. Calculer la couverture pour chaque nœud prioritaire
def calculate_node_coverages(
    G: nx.Graph,
    prioritized_nodes: set,
    vanne_df: pd.DataFrame,
    df: pd.DataFrame,
    max_distance: float,
) -> dict:
    """Calcule la couverture pour chaque nœud prioritaire en tenant compte des statuts de nœud."""

    # Validation de max_distance
    if max_distance <= 0:
        raise ValueError(
            "La distance maximale de couverture doit être supérieure à zéro."
        )

    # Identifier les nœuds FORCÉ à partir du DataFrame
    forced_nodes = set(vanne_df[vanne_df["FORCE"]]["ID_NOEUD"])

    # Créer le mapping des arêtes vers les identifiants de canalisations
    edge_to_id = {
        (row["ID_NOEUD_1"], row["ID_NOEUD_2"]): row["ID_CANA"]
        for _, row in df.iterrows()
        if (row["ID_NOEUD_1"], row["ID_NOEUD_2"]) in G.edges()
    }

    # Calcul de la couverture pour chaque nœud prioritaire
    node_coverage = {
        node: compute_covered_pipelines(G, node, edge_to_id, max_distance)
        for node in prioritized_nodes
    }

    return node_coverage, forced_nodes


# 6. Placement glouton des prélocalisateurs
def greedy_node_selection(
    node_coverage: dict, n: int, forced_nodes: set
) -> tuple[set, set]:
    """
    Sélectionne les N meilleurs nœuds pour maximiser la couverture, en tenant compte des nœuds FORCÉ.
    """
    selected_nodes = set(
        forced_nodes
    )  # Commencer avec les nœuds FORCÉ déjà sélectionnés
    covered_ids = set()

    # Mettre à jour la couverture initiale avec les nœuds FORCÉ
    for node in forced_nodes:
        covered_ids.update(node_coverage.get(node, set()))

    # Limiter n au nombre de nœuds disponibles
    n = min(n, len(node_coverage))

    # Sélection gloutonne en excluant les nœuds FORCÉ déjà ajoutés
    for _ in range(n - len(forced_nodes)):
        best_node = max(
            (node for node in node_coverage if node not in selected_nodes),
            key=lambda node: len(node_coverage[node] - covered_ids),
            default=None,
        )

        if best_node is None:
            break

        selected_nodes.add(best_node)
        covered_ids.update(node_coverage[best_node])

    return selected_nodes, covered_ids


# 7. Calculer le pourcentage de couverture
def calculate_coverage_percentage(total_ids: set, covered_ids: set) -> float:
    """Calcule le pourcentage de canalisations couvertes."""

    total_ids = set(total_ids)  # Convert to set if not already
    covered_ids = set(covered_ids)  # Convert to set if not already

    if not total_ids:
        return 0.0  # Handle division by zero
    return (len(covered_ids) / len(total_ids)) * 100
