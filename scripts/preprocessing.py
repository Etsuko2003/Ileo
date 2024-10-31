import pandas as pd
from pyproj import Transformer
from scripts.constants import *

# Initialize the transformer once
transformer = Transformer.from_crs(2154, 4326, always_xy=True)


# Function to convert Lambert-93 coordinates to GPS decimal
def lambert93_to_gps(x: float, y: float) -> tuple:
    longitude, latitude = transformer.transform(x, y)
    return latitude, longitude


def canalisation_with_latitude_longitude(cana_df) -> pd.DataFrame:
    """Charge les données des canalisations et ajoute les coordonnées GPS."""

    # Attempt to read the Excel file and handle possible errors
    try:
        with pd.ExcelFile(NOEUD_DATA_PATH) as noeud_file:
            noeud_df = noeud_file.parse()
    except FileNotFoundError:
        raise FileNotFoundError(f"Excel file not found at path: {NOEUD_DATA_PATH}")

    # Apply the conversion to Lambert-93 coordinates
    noeud_df[["LONGITUDE", "LATITUDE"]] = noeud_df.apply(
        lambda row: lambert93_to_gps(row["X"], row["Y"]), axis=1, result_type="expand"
    )

    # Join coordinates for ID_NOEUD_1
    cana_df = cana_df.merge(
        noeud_df, how="left", left_on="ID_NOEUD_1", right_on="ID_NOEUD"
    )
    if "LATITUDE" in noeud_df.columns and "LONGITUDE" in noeud_df.columns:
        cana_df.rename(
            columns={"LATITUDE": "LATITUDE_1", "LONGITUDE": "LONGITUDE_1"}, inplace=True
        )
    else:
        raise KeyError("LATITUDE or LONGITUDE column missing in noeud_df.")

    # Join coordinates for ID_NOEUD_2
    cana_df = cana_df.merge(
        noeud_df, how="left", left_on="ID_NOEUD_2", right_on="ID_NOEUD"
    )
    cana_df.rename(
        columns={"LATITUDE": "LATITUDE_2", "LONGITUDE": "LONGITUDE_2"}, inplace=True
    )

    # Check for missing coordinates
    if (
        cana_df[["LATITUDE_1", "LONGITUDE_1", "LATITUDE_2", "LONGITUDE_2"]]
        .isnull()
        .any()
        .any()
    ):
        raise ValueError("Some ID_NOEUD values could not be matched in noeud_df.")

    # Keep only the necessary columns
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

    # Reset index to avoid potential issues in later processing
    cana_df.reset_index(drop=True, inplace=True)

    return cana_df


def create_columns(vanne_df: pd.DataFrame) -> pd.DataFrame:
    # Créer une copie du DataFrame d'entrée
    vanne_df = vanne_df.copy()

    # Ajouter les colonnes avec des valeurs par défaut
    vanne_df["BLOQUE"] = False
    vanne_df["BLOQUE"] = vanne_df["BLOQUE"].astype(bool)
    vanne_df["FORCE"] = False
    vanne_df["LOCALISATION"] = ""

    return vanne_df


def combine_dataframe(vanne_df: pd.DataFrame, gestion_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine two DataFrames by merging and updating specified columns.

    Parameters:
    vanne_df (pd.DataFrame): DataFrame containing vanne data.
    gestion_df (pd.DataFrame): DataFrame containing gestion data with updates.

    Returns:
    pd.DataFrame: A DataFrame containing the combined data with updated columns.
    """
    required_cols_vanne = ["ID_NOEUD", "ID_VANNE"]
    required_cols_gestion = ["ID_NOEUD", "ID_VANNE", "BLOQUE", "FORCE", "LOCALISATION"]

    for col in required_cols_vanne:
        if col not in vanne_df.columns:
            raise ValueError(f"{col} is missing from vanne_df")

    for col in required_cols_gestion:
        if col not in gestion_df.columns:
            raise ValueError(f"{col} is missing from gestion_df")

    # Jointure gauche pour combiner les deux DataFrames
    merged_df = pd.merge(
        vanne_df,
        gestion_df,
        on=["ID_NOEUD", "ID_VANNE"],
        how="left",
        suffixes=("", "_gestion"),
    )

    # Mettre à jour les colonnes avec des valeurs de gestion_df
    for col in ["BLOQUE", "FORCE", "LOCALISATION"]:
        gestion_col = col + "_gestion"
        if gestion_col in merged_df.columns:
            merged_df[col] = merged_df[gestion_col].combine_first(merged_df[col])

    # Supprimer les colonnes temporaires
    columns_to_drop = [
        f"{col}_gestion"
        for col in [
            "BLOQUE",
            "FORCE",
            "LOCALISATION",
            "DIAMETRE",
            "ID_CANA_1",
            "ID_CANA_2",
        ]
    ]
    columns_to_drop = [col for col in columns_to_drop if col in merged_df.columns]

    merged_df.drop(columns=columns_to_drop, inplace=True)

    return merged_df
