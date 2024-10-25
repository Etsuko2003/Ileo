import pandas as pd
from pyproj import Transformer

from scripts.constants import *


# Fonction pour convertir les coordonnées Lambert-93 en GPS décimal
def lambert93_to_gps(x: float, y: float) -> tuple:
    transformer = Transformer.from_crs(2154, 4326, always_xy=True)
    longitude, latitude = transformer.transform(x, y)
    return latitude, longitude


def canalisation_with_latitude_longitude() -> pd.DataFrame:
    """Charge les données des canalisations et ajoute les coordonnées GPS."""
    # Lecture des fichiers Excel
    with pd.ExcelFile(CANA_DATA_PATH) as cana_file, pd.ExcelFile(
        NOEUD_DATA_PATH
    ) as noeud_file:
        cana_df = cana_file.parse()
        noeud_df = noeud_file.parse()

    # Appliquer la conversion aux colonnes de coordonnées Lambert-93
    noeud_df[["LONGITUDE", "LATITUDE"]] = noeud_df.apply(
        lambda row: lambert93_to_gps(row["X"], row["Y"]), axis=1, result_type="expand"
    )

    # Joindre les coordonnées pour ID_NOEUD_1
    cana_df = cana_df.merge(
        noeud_df, how="left", left_on="ID_NOEUD_1", right_on="ID_NOEUD"
    )
    cana_df.rename(
        columns={"LATITUDE": "LATITUDE_1", "LONGITUDE": "LONGITUDE_1"}, inplace=True
    )

    # Joindre les coordonnées pour ID_NOEUD_2
    cana_df = cana_df.merge(
        noeud_df, how="left", left_on="ID_NOEUD_2", right_on="ID_NOEUD"
    )
    cana_df.rename(
        columns={"LATITUDE": "LATITUDE_2", "LONGITUDE": "LONGITUDE_2"}, inplace=True
    )

    # Garder seulement les colonnes nécessaires
    cana_df = cana_df[
        [
            "ID_CANA",
            "ID_NOEUD_1",
            "ID_NOEUD_2",
            "LONGUEUR_EN_M",
            "DIAMETRE",
            "COMMUNE",
            "MATERIAU",
            "LATITUDE_1",
            "LONGITUDE_1",
            "LATITUDE_2",
            "LONGITUDE_2",
        ]
    ]

    return cana_df
