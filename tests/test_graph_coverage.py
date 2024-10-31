import pytest
from pytest import approx
import pandas as pd
import networkx as nx
from scripts.constants import MAX_DIAMETER_TO_CONSIDER
from scripts.graph_coverage import (
    create_graph,
    filter_vanne_nodes,
    get_prioritized_nodes,
    compute_covered_pipelines,
    calculate_node_coverages,
    greedy_node_selection,
    calculate_coverage_percentage,
)


@pytest.fixture
def sample_df():
    """Fixture pour créer un DataFrame de test."""
    return pd.DataFrame(
        {
            "ID_NOEUD_1": ["node1", "node2", "node1"],
            "ID_NOEUD_2": ["node2", "node3", "node3"],
            "LATITUDE_1": [0, 1, 0],
            "LONGITUDE_1": [0, 1, 0],
            "LATITUDE_2": [1, 1, 1],
            "LONGITUDE_2": [1, 1, 1],
            "LONGUEUR_EN_M": [100, 200, 150],
            "ID_CANA": ["c1", "c2", "c3"],
            "DIAMETRE": [300, 400, 350],
        }
    )


@pytest.fixture
def sample_graph(sample_df):
    """Fixture pour créer un graphe à partir d'un DataFrame."""
    return create_graph(sample_df)


@pytest.fixture
def sample_vanne_df():
    """Fixture pour créer un DataFrame de vannes de test."""
    return pd.DataFrame(
        {
            "ID_NOEUD": ["node1", "node2"],
            "FORCE": [False, True],
            "BLOQUE": [False, False],
        }
    )


def test_create_graph(sample_df):
    """Test de la fonction create_graph."""
    G = create_graph(sample_df)
    assert len(G.nodes) == 3
    assert len(G.edges) == 3


def test_filter_vanne_nodes(sample_graph, sample_vanne_df):
    """Test de la fonction filter_vanne_nodes."""
    vanne_nodes = filter_vanne_nodes(sample_graph, sample_vanne_df)
    assert vanne_nodes == {"node1", "node2"}


def test_get_prioritized_nodes(sample_graph, sample_vanne_df):
    """Test de la fonction get_prioritized_nodes."""
    vanne_nodes = filter_vanne_nodes(sample_graph, sample_vanne_df)
    prioritized_nodes = get_prioritized_nodes(
        sample_graph, vanne_nodes, sample_vanne_df
    )
    assert prioritized_nodes == {"node1", "node2"}


def test_compute_covered_pipelines(sample_graph):
    """Test de la fonction compute_covered_pipelines."""
    edge_to_id = {
        ("node1", "node2"): "c1",
        ("node2", "node3"): "c2",
        ("node1", "node3"): "c3",
    }
    covered_ids = compute_covered_pipelines(sample_graph, "node1", edge_to_id, 200)
    assert covered_ids == {
        "c1",
        "c3",
    }  # Assumes node1 can cover these edges within distance


def test_calculate_node_coverages(sample_graph, sample_vanne_df, sample_df):
    """Test de la fonction calculate_node_coverages."""
    vanne_nodes = filter_vanne_nodes(sample_graph, sample_vanne_df)
    node_coverages, forced_nodes = calculate_node_coverages(
        sample_graph, vanne_nodes, sample_vanne_df, sample_df, 200
    )
    assert node_coverages
    assert forced_nodes == {"node2"}


def test_greedy_node_selection():
    """Test de la fonction greedy_node_selection."""
    node_coverage = {
        "node1": {"c1"},
        "node2": {"c2"},
    }
    selected_nodes, covered_ids = greedy_node_selection(node_coverage, 1, {"node2"})
    assert selected_nodes == {"node2"}
    assert "c2" in covered_ids


def test_calculate_coverage_percentage():
    """Test de la fonction calculate_coverage_percentage."""
    total_ids = {"c1", "c2", "c3"}
    covered_ids = {"c1", "c2"}
    percentage = calculate_coverage_percentage(total_ids, covered_ids)
    assert approx(percentage, rel=1e-2) == 66.67
