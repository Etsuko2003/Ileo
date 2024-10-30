# constants.py

# Chemins des fichiers de données
VANNE_DATA_PATH = "data/VANNE.xlsx"
CANA_DATA_PATH = "data/CANALISATION.xlsx"
NOEUD_DATA_PATH = "data/NOEUD.xlsx"
CAN_DATA_PATH = "data/can.xlsx"
GESTION_DATA_PATH = "data/GESTION.xlsx"

# Valeurs par défaut
DEFAULT_COVERAGE_DISTANCE = 200.0
MIN_PRELOCATORS = 60.0
MAX_LENGTH_THRESHOLD = 30000.0
MAX_DIAMETER_TO_CONSIDER = 350

# Messages d'entrée utilisateur
PROMPT_COMMUNE_SELECTION = (
    "Entrez le.s numéro.s correspondant.s aux communes (séparés par des virgules) : "
)
PROMPT_NUM_PRELOCATORS_UNDER_THRESHOLD = "La somme des longueurs est inférieure à 30 000 m. Combien de prélocalisateurs voulez-vous déployer ? "
PROMPT_NUM_PRELOCATORS_OVER_THRESHOLD = "La somme des longueurs est supérieure à 30 000 m.  Combien de prélocalisateurs voulez-vous déployer ? "
