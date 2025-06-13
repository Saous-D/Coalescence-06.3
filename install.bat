@echo off
REM Vérifiez si Python est installé
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python n'est pas installé. Veuillez installer Python avant de continuer.
    exit /b 1
)


REM Définissez le chemin du projet et le nom de l'environnement virtuel
set "PROJECT_PATH=."
set "VENV_NAME=venv"

REM Affichez les variables pour le débogage
echo Chemin du projet: %PROJECT_PATH%
echo Nom de l'environnement virtuel: %VENV_NAME%

REM Changez le répertoire de travail au chemin du projet
cd /d %PROJECT_PATH%
IF %ERRORLEVEL% NEQ 0 (
    echo Impossible de changer de répertoire vers %PROJECT_PATH%. Veuillez vérifier le chemin.
    exit /b 1
)

REM Vérifiez si l'environnement virtuel existe déjà
IF EXIST %VENV_NAME% (
    echo L'environnement virtuel existe déjà.
) ELSE (
    echo Création de l'environnement virtuel...
    py -3.9 -m venv %VENV_NAME%
    IF %ERRORLEVEL% NEQ 0 (
        echo Échec de la création de l'environnement virtuel.
        exit /b 1
    )
)

REM Vérifiez si le script d'activation existe
IF NOT EXIST %VENV_NAME%\Scripts\activate.bat (
    echo Le script d'activation %VENV_NAME%\Scripts\activate.bat n'existe pas.
    exit /b 1
)

REM Lancez une nouvelle fenêtre de commande avec l'environnement virtuel activé, installez les dépendances et lancez app.py
echo Activation de l'environnement virtuel, installation des dépendances et lancement de app.py...
start cmd /k "%VENV_NAME%\Scripts\activate.bat && python -m pip install --upgrade pip setuptools wheel && pip install -r requirements.txt && python app.py"

REM Fin du script principal
exit /b 0
