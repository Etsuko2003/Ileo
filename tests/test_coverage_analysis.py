import pytest
import pandas as pd
from math import floor
from scripts.constants import (
    MAX_LENGTH_THRESHOLD,
    MIN_PRELOCATORS,
)  # Assure-toi que ces constantes sont définies
from scripts.coverage_analysis import (
    display_commune_options,
    filter_data_by_commune,
    calculate_total_length,
    get_max_prelocators,
    generate_coverage_plot,
    save_selected_nodes_to_excel,
)

# Mock pour Streamlit
import streamlit as st
from unittest.mock import patch


@pytest.fixture
def sample_data():
    """Fixture pour créer un DataFrame de test"""
    return pd.DataFrame(
        {
            "COMMUNE": ["Commune A", "Commune B", "Commune A", "Commune C"],
            "LONGUEUR_EN_M": [100, 200, 150, 50],
        }
    )


def test_display_commune_options():
    """Test de la fonction display_commune_options"""
    with patch("streamlit.selectbox") as mock_selectbox:
        mock_selectbox.return_value = "Commune A"
        result = display_commune_options(["Commune A", "Commune B"])
        mock_selectbox.assert_called_once()
        assert result == "Commune A"


def test_filter_data_by_commune(sample_data):
    """Test de la fonction filter_data_by_commune"""
    result = filter_data_by_commune(sample_data, ["Commune A"])
    assert len(result) == 2
    assert all(result["COMMUNE"] == "Commune A")


def test_calculate_total_length(sample_data):
    """Test de la fonction calculate_total_length"""
    commune_data = sample_data[sample_data["COMMUNE"] == "Commune A"]
    total_length = calculate_total_length(commune_data)
    assert total_length == 250  # 100 + 150


def test_get_max_prelocators():
    """Test de la fonction get_max_prelocators"""
    total_length = 200  # Exemple
    max_prelocators = get_max_prelocators(total_length)
    expected = floor(total_length / MAX_LENGTH_THRESHOLD * MIN_PRELOCATORS)
    assert max_prelocators == expected


def test_generate_coverage_plot():
    """Test de la fonction generate_coverage_plot (simulé ici)"""
    with patch("streamlit.subheader") as mock_subheader, patch(
        "streamlit.line_chart"
    ) as mock_line_chart, patch("streamlit.write") as mock_write:
        generate_coverage_plot([1, 2, 3], [10.0, 20.0, 30.0], 3, "Commune A")
        mock_subheader.assert_called_once()
        mock_line_chart.assert_called_once()
        assert mock_write.call_count == 2  # On attend 2 appels à st.write


def test_save_selected_nodes_to_excel(sample_data):
    """Test de la fonction save_selected_nodes_to_excel (simulé ici)"""
    selected_nodes = ["node1", "node2"]
    commune_name = "Commune A"
    vanne_df = pd.DataFrame(
        {
            "ID_NOEUD": ["node1", "node2", "node3"],
            "AUTRE_INFO": ["info1", "info2", "info3"],
        }
    )

    with patch("pandas.DataFrame.to_excel") as mock_to_excel:
        save_selected_nodes_to_excel(selected_nodes, commune_name, vanne_df)
        mock_to_excel.assert_called_once_with(
            f"data/created/{commune_name}_noeud_selected.xlsx"
        )
