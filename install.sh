#!/bin/bash

# Vérifiez si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "Python n'est pas installé. Veuillez installer Python avant de continuer."
    exit 1
fi

# Définissez le chemin du projet et le nom de l'environnement virtuel
PROJECT_PATH="."
VENV_NAME="venv"

# Affichez les variables pour le débogage
echo "Chemin du projet : $PROJECT_PATH"
echo "Nom de l'environnement virtuel : $VENV_NAME"

# Changez le répertoire de travail au chemin du projet
cd "$PROJECT_PATH" || { echo "Impossible de changer de répertoire vers $PROJECT_PATH. Veuillez vérifier le chemin."; exit 1; }

# Vérifiez si l'environnement virtuel existe déjà
if [ -d "$VENV_NAME" ]; then
    echo "L'environnement virtuel existe déjà."
else
    echo "Création de l'environnement virtuel..."
    python3 -m venv "$VENV_NAME"
    if [ $? -ne 0 ]; then
        echo "Échec de la création de l'environnement virtuel."
        exit 1
    fi
fi

# Activez l'environnement virtuel
source "$VENV_NAME/bin/activate"
if [ $? -ne 0 ]; then
    echo "Le script d'activation $VENV_NAME/bin/activate n'existe pas."
    exit 1
fi

# Installez les dépendances et lancez app.py
echo "Installation des dépendances et lancement de app.py..."
pip install -r requirements.txt
python app.py

# Fin du script principal
deactivate
