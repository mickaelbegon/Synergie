# Synergie Data

Logiciel de reconnaissance de sauts en patinage artistique a partir de donnees IMU Movella DOT.

Le depot contient deux usages principaux :

- une application graphique pour la collecte et l'export des capteurs
- une CLI pour l'entrainement et le retraitement des fichiers CSV
- une petite interface graphique utilitaire pour les workflows de dev et d'analyse

## Installation

### Environnement Conda

Le fichier [environment.yml](C:\Users\micka\Documents\GIT\Synergie_Data\Synergie_Data\environment.yml) decrit l'environnement conseille pour le developpement et les tests.

```sh
conda env create -f environment.yml
conda activate synergie-data
```

Si l'environnement existe deja et que tu veux le remettre a jour a partir du fichier :

```sh
conda env update -n synergie-data -f environment.yml --prune
conda activate synergie-data
```

Verification rapide conseillee :

```sh
python -c "import tensorflow as tf; print(tf.__version__)"
python -m unittest tests.test_operations
```

Sous Windows, si `conda activate` n'est pas reconnu dans PowerShell, ouvre d'abord un terminal Anaconda/Miniconda ou utilise les chemins complets :

```powershell
& "C:\Users\micka\miniconda3\Scripts\conda.exe" env create -f environment.yml
& "C:\Users\micka\miniconda3\envs\synergie-data\python.exe" -c "import tensorflow as tf; print(tf.__version__)"
& "C:\Users\micka\miniconda3\envs\synergie-data\python.exe" tools_gui.py
```

Les sessions de collecte ne sont plus codees dans `constants.py`.
Elles sont maintenant stockees dans [config/sessions.json](C:\Users\micka\Documents\GIT\Synergie_Data\Synergie_Data\config\sessions.json) et peuvent etre ajoutees depuis l'onglet `Sessions` de `tools_gui.py`.

### Dependances externes

L'application GUI depend aussi :

- du SDK Movella DOT
- d'un fichier de credentials Firebase local non versionne

Le SDK Movella est disponible ici : [Movella DOT software documentation](https://www.movella.com/support/software-documentation)

Compatibilite Movella:

- le code essaye maintenant automatiquement les bindings PC SDK Python `py311`, `py310`, puis `py39`
- la release officielle Movella DOT `2023.6.0` a introduit la famille `Movella DOT` et le PC SDK associe
- d'apres Movella, le firmware `3.0.0+` doit etre utilise avec les SDK/apps `2023.6.0+`
- le Data Exporter recent reste distinct selon le seuil firmware `2.4.0`

## Lancer l'application

```sh
python app.py
python tools_gui.py
```

Flux principal :

- connexion d'un coach via Firebase
- detection des capteurs Bluetooth et USB
- demarrage d'un enregistrement au debranchement
- arret et export au rebranchement
- prediction automatique des sauts apres export

Dans `tools_gui.py`, l'onglet `Inspect IMU` permet aussi :

- d'afficher les signaux utilises pour localiser les sauts
- d'ajuster les sliders de detection
- de selectionner un saut pour afficher un zoom dedie
- de lire le dossier de session choisi et lister tous les fichiers disponibles avec infos de base sur le fichier selectionne

Pipeline conseille pour ajouter des donnees d'entrainement dans `tools_gui.py` :

1. `Data - New` : choisir une seance brute depuis `data/new`.
2. Le GUI propose automatiquement un identifiant de session et un chemin cible dans `data/raw`; `Data - Sessions` sert surtout a verifier ou corriger les exceptions.
3. `Data - New` : utiliser `Process for annotation`.
4. Le traitement cree un CSV `*_for_annotation.csv`, des segments IMU, et pre-remplit si possible `type`, `success` et `turns` avec les modeles selectionnes.
5. `Data - Annotate` : verifier les propositions, synchroniser la video, corriger les labels et finaliser le fichier annote.
6. `Review - Quality` et `Review - Detection` : verifier les outliers, faux positifs et faux negatifs avant re-entrainement.
7. `Models - Train` : re-entrainer les modeles quand le dataset a change.
8. `Models - Audit`, `Models - Importance` et `Models - Tune` : verifier les nouveaux modeles avant de les reutiliser comme solution initiale.

`Data - Process` reste utile pour retraiter manuellement ou en batch des CSV deja classes dans `data/raw`, mais n'est pas obligatoire dans le flux principal `new -> annotate -> train`.

L'onglet `Annotate` permet maintenant aussi de :

- charger une video de la seance en plus du CSV `for_annotation`
- memoriser un offset de synchronisation par capteur IMU
- relire rapidement (`x5`) les 5 secondes qui precedent le saut courant jusqu'a son instant d'apparition estime
- afficher dynamiquement l'heure video recalculee pour chaque saut selon l'offset du capteur concerne

L'onglet `Train` permet maintenant aussi de choisir explicitement l'architecture d'entrainement selon la tache:

- `type` : `inceptiontime` ou `transformer`
- `success` : `tcn` ou `lstm`

## CLI

La CLI est organisee par sous-commandes :

```sh
python main.py list-sessions
python main.py show-session 1331
python main.py list-session-files 1331
python main.py train type --epochs 10
python main.py train type --architecture inceptiontime --epochs 10
python main.py train success --epochs 20
python main.py train success --architecture tcn --epochs 20
python main.py benchmark type --model summary
python main.py benchmark type --model minirocket
python main.py benchmark type --model hydra
python main.py process-file data/raw/0406/0927/1_D422CD0076F7_20240604_092734.csv --output data/pending/example_predictions.csv
python main.py repredict
```

Compatibilite conservee :

```sh
python main.py -t type
python main.py -repredict
```

## Structure du projet

- `app.py` : point d'entree de l'application graphique
- `main.py` : point d'entree CLI
- `tools_gui.py` : petite interface graphique utilitaire pour la CLI
- `synergie/cli.py` : parsing des commandes
- `synergie/config.py` : constantes partagees pour fenetres et seuils
- `synergie/operations.py` : facade de compatibilite pour les workflows GUI/CLI
- `synergie/session_store.py` : lecture/ecriture des sessions dans le fichier JSON
- `synergie/services/` : logique metier reutilisable hors interface
- `synergie/tool_gui.py` : interface graphique simple pour les workflows hors capteurs
- `core/` : logique metier, traitement de donnees, modeles, acces base de donnees
- `front/` : composants interface Tkinter/ttkbootstrap
- `tests/` : tests legers de non-regression pour la CLI, l'algo IMU et les services

## Tests

Une premiere base de tests unitaires est fournie :

```sh
python -m unittest discover -s tests
```

Ces tests ne couvrent pas encore la partie modele ni la connexion aux capteurs. Ils valident surtout la structure CLI et quelques invariants de configuration.

## Architecture des services

La logique applicative a ete progressivement extraite de `synergie/operations.py` vers des services specialises :

- `session_service.py` : sessions configurees, chemins et fichiers de sortie libres
- `new_data_service.py` : decouverte des nouvelles donnees IMU
- `annotation_generation_service.py` : generation des segments et CSV `for_annotation`
- `annotation_service.py` : metadonnees, statuts et utilitaires d'annotation
- `annotation_finalization_service.py` : archivage et fusion des labels vers le dataset d'entrainement
- `csv_processing_service.py` : traitement CSV brut vers jumplist
- `prediction_service.py` : predictions type/succes reutilisables
- `quality_service.py` et `detection_tuning_service.py` : controle qualite et reglage des seuils
- `signal_importance_service.py` : importance des signaux temporels, scalaires et fenetres
- `training_dataset_service.py` : statistiques, doublons et etat du dataset
- `training_service.py` : entrainement, tuning et promotion des modeles
- `model_registry_service.py` : registre et audit des modeles disponibles
- `video_service.py` : videos de seance et rapprochement temporel

Cette separation permet de garder `operations.py` comme une facade stable tout en rendant chaque domaine testable independamment.

## Modeles IA

Le projet garde TensorFlow/Keras comme base de production, mais la CLI permet maintenant de tester plusieurs architectures plus adaptees aux series temporelles IMU :

- `type` : `transformer` ou `inceptiontime`
- `success` : `lstm` ou `tcn`

Les architectures recommandees par defaut sont maintenant :

- `inceptiontime` pour la classification du type de saut
- `tcn` pour la prediction de la reussite

L'environnement inclut aussi `aeon` pour preparer des essais de classifieurs de series temporelles rapides comme MiniRocket/Hydra sur CPU.

La commande `benchmark` sert a comparer rapidement des approches sur le meme dataset annote :

- `--model summary` : resume du dataset exploitable
- `--model minirocket` : benchmark CPU `MiniRocket + RidgeClassifierCV`
- `--model hydra` : benchmark CPU `HydraClassifier`

## Branche `pariterre`

Une analyse de la branche distante `pariterre/main` est documentee dans [docs/pariterre_merge.md](C:\Users\micka\Documents\GIT\Synergie_Data\Synergie_Data\docs\pariterre_merge.md).

L'audit de l'algorithme de reconnaissance des sauts IMU est documente dans [docs/imu_jump_algorithm_review.md](C:\Users\micka\Documents\GIT\Synergie_Data\Synergie_Data\docs\imu_jump_algorithm_review.md).

Resume :

- la branche contient une refonte large vers un package `synergie/`
- elle apporte de bonnes idees de structure et d'environnement
- elle est trop intrusive pour etre fusionnee en bloc sans validation materielle

## Donnees

Le modele actuel s'appuie sur un jeu de donnees annote prive. Les dossiers `data/raw`, `data/new` et `data/annotated` contiennent des artefacts de travail qui peuvent etre volumineux et sensibles.

## Credits

Projet realise par S2M pour Patinage Quebec.
