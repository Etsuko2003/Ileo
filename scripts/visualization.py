import folium
import networkx as nx


def plot_graph_on_folium(G: nx.Graph, commune_name: str = "All") -> None:
    """Affiche le graphe des canalisations sur une carte Folium."""
    # Récupération des positions des nœuds
    positions = nx.get_node_attributes(G, "pos")

    # Création de la carte centrée sur un des nœuds
    latitudes = [pos[0] for pos in positions.values()]
    longitudes = [pos[1] for pos in positions.values()]
    center_lat = sum(latitudes) / len(latitudes)
    center_lon = sum(longitudes) / len(longitudes)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # Ajout des nœuds (en bleu)
    for node, pos in positions.items():
        folium.CircleMarker(
            location=[pos[0], pos[1]],
            radius=5,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.7,
            popup=f"Noeud: {node}",
        ).add_to(m)

    # Ajout des arêtes (canalisations, en vert)
    for edge in G.edges():
        pos1 = positions[edge[0]]
        pos2 = positions[edge[1]]
        folium.PolyLine(
            locations=[(pos1[0], pos1[1]), (pos2[0], pos2[1])],
            color="green",
            weight=2,
            opacity=0.7,
        ).add_to(m)

    m.save(f"graphe_canalisation_{commune_name}.html")


def plot_solution_on_folium(
    G: nx.Graph, selected_nodes: set, covered_ids: set, vanne_nodes: set
) -> folium.Map:
    """Affiche le réseau entier avec les nœuds et canalisations couverts en couleur."""
    # Centrage de la carte autour des positions des nœuds
    first_node = next(iter(G.nodes()))
    latitude, longitude = G.nodes[first_node]["pos"]
    m = folium.Map(location=[latitude, longitude], zoom_start=13)

    # Ajout de tous les nœuds (en bleu)
    for node, pos in nx.get_node_attributes(G, "pos").items():
        folium.CircleMarker(
            location=pos, radius=5, color="blue", fill=True, tooltip=f"{node}"
        ).add_to(m)

    # Ajout des nœuds de vanne (en orange)
    for node in vanne_nodes:
        pos = G.nodes[node]["pos"]
        folium.CircleMarker(
            location=pos, radius=5, color="orange", fill=True, tooltip=f"{node}"
        ).add_to(m)

    # Ajout de tous les liens (canalisations)
    for u, v, data in G.edges(data=True):
        lat1, lon1 = G.nodes[u]["pos"]
        lat2, lon2 = G.nodes[v]["pos"]

        # Vérification du diamètre pour déterminer la couleur
        color = "gray" if data["diameter"] > 350 else "yellow"

        folium.PolyLine(
            locations=[(lat1, lon1), (lat2, lon2)], color=color, weight=3
        ).add_to(m)

    # Mise en évidence des nœuds sélectionnés (en rouge)
    for node in selected_nodes:
        pos = G.nodes[node]["pos"]
        folium.CircleMarker(
            location=pos, radius=5, color="red", fill=True, tooltip=f"ID: {node}"
        ).add_to(m)

    # Mise en évidence des canalisations couvertes (en vert)
    for u, v in G.edges():
        if G.edges[u, v]["ID_CANA"] in covered_ids:
            lat1, lon1 = G.nodes[u]["pos"]
            lat2, lon2 = G.nodes[v]["pos"]
            folium.PolyLine(
                locations=[(lat1, lon1), (lat2, lon2)], color="green", weight=3
            ).add_to(m)

    return m
