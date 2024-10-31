import pytest
import pandas as pd
from scripts.constants import NOEUD_DATA_PATH
from scripts.preprocessing import (
    lambert93_to_gps,
    canalisation_with_latitude_longitude,
    create_columns,
    combine_dataframe,
)


# Exemple de DataFrame pour les tests
@pytest.fixture
def sample_noeud_df():
    """Fixture pour un DataFrame de noeuds."""
    return pd.DataFrame(
        {
            "ID_NOEUD": ["node1", "node2", "node3"],
            "X": [600000, 600100, 600200],
            "Y": [100000, 100100, 100200],
        }
    )


@pytest.fixture
def sample_cana_df(sample_noeud_df):
    """Fixture pour un DataFrame de canalisations."""
    return pd.DataFrame(
        {
            "ID_CANA": ["c1", "c2"],
            "ID_NOEUD_1": ["node1", "node2"],
            "ID_NOEUD_2": ["node2", "node3"],
            "LONGUEUR_EN_M": [100, 200],
            "DIAMETRE": [300, 400],
            "COMMUNE": ["CommuneA", "CommuneB"],
            "MATERIAU": ["PVC", "Acier"],
        }
    )


@pytest.fixture
def sample_vanne_df():
    """Fixture pour un DataFrame de vannes."""
    return pd.DataFrame({"ID_NOEUD": ["node1", "node2"], "ID_VANNE": ["v1", "v2"]})


@pytest.fixture
def sample_gestion_df():
    """Fixture pour un DataFrame de gestion."""
    return pd.DataFrame(
        {
            "ID_NOEUD": ["node1", "node2"],
            "ID_VANNE": ["v1", "v2"],
            "BLOQUE": [True, False],
            "FORCE": [False, True],
            "LOCALISATION": ["loc1", "loc2"],
        }
    )


def test_lambert93_to_gps():
    """Test de la fonction lambert93_to_gps."""
    x, y = 600000, 100000
    latitude, longitude = lambert93_to_gps(x, y)
    assert isinstance(latitude, float)
    assert isinstance(longitude, float)


def test_create_columns():
    """Test de la fonction create_columns."""
    vanne_df = pd.DataFrame({"ID_NOEUD": ["node1", "node2"]})
    result = create_columns(vanne_df)
    assert "BLOQUE" in result.columns
    assert "FORCE" in result.columns
    assert result["BLOQUE"].dtype == bool
    assert result["FORCE"].dtype == bool


def test_combine_dataframe(sample_vanne_df, sample_gestion_df):
    """Test de la fonction combine_dataframe."""
    result = combine_dataframe(sample_vanne_df, sample_gestion_df)
    assert result["BLOQUE"].iloc[0] == True
    assert result["FORCE"].iloc[0] == False
    assert result["LOCALISATION"].iloc[0] == "loc1"
