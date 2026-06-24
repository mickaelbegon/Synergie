# Synergie Data

Logiciel de reconnaissance de sauts en patinage artistique a partir de donnees IMU Movella DOT.

Le depot contient deux usages principaux :

- une application graphique pour la collecte et l'export des capteurs
- une CLI pour l'entrainement et le retraitement des fichiers CSV
- une petite interface graphique utilitaire pour les workflows de dev et d'analyse

## Installation

### Environnement Conda

Le fichier [environment.yml](C:\Users\micka\Documents\GIT\Synergie_Data\Synergie_Data\environment.yml) decrit l'environnement conseille pour Windows, le developpement et les tests.

```sh
conda env create -f environment.yml
conda activate synergie-data
```

Sur Mac Intel (`osx-64`), utiliser plutot [environment-macos-intel.yml](C:\Users\micka\Documents\GIT\Synergie_Data\Synergie_Data\environment-macos-intel.yml). TensorFlow `2.20.0` n'est pas publie pour cette plateforme, donc cet environnement utilise TensorFlow `2.16.2` et retire les dependances Windows `winrt-*`.

```sh
conda env create -f environment-macos-intel.yml
conda activate synergie-data
```

Si une creation Mac Intel a echoue pendant l'installation `pip`, repartir d'un environnement propre avant de relancer :

```sh
conda env remove -n synergie-data
conda env create -f environment-macos-intel.yml
conda activate synergie-data
```

L'environnement Mac installe `cryptography`, `openssl` et `pkg-config` via Conda pour eviter que `pip` tente de compiler `cryptography` localement.

Si l'environnement existe deja et que tu veux le remettre a jour a partir du fichier :

```sh
conda env update -n synergie-data -f environment.yml --prune
conda activate synergie-data
```

Sur Mac Intel :

```sh
conda env update -n synergie-data -f environment-macos-intel.yml --prune
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

Pour contribuer au refactor de l'interface sans casser les workflows d'annotation, voir [docs/gui_refactor_architecture.md](C:\Users\micka\Documents\GIT\Synergie_Data\Synergie_Data\docs\gui_refactor_architecture.md).

### Dependances externes

L'application GUI depend aussi :

- du SDK Movella DOT
- d'un fichier de credentials Firebase local non versionne
- de `ffprobe` pour lire certaines metadonnees video, si disponible
- de `ffmpeg` pour creer une copie video optimisee et rendre la lecture plus fluide dans `Data - Annotate`

Sur Mac Intel, `tools_gui.py` peut servir au traitement, a l'annotation et a l'entrainement, mais la collecte capteurs via Movella/Windows Bluetooth n'est pas couverte par `environment-macos-intel.yml`.

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

`app.py` et `tools_gui.py` n'ont pas le meme role :

- `app.py` : interface principale de collecte terrain avec Firebase, capteurs et export
- `tools_gui.py` : interface utilitaire de dev, de retraitement, d'annotation, d'audit qualite et d'entrainement

Sous Windows, le plus simple pour ouvrir l'outil utilitaire avec le bon environnement est :

```powershell
& "C:\Users\micka\Documents\GIT\Synergie_Data\Synergie_Data\launch_tools_gui.bat"
```

### `app.py` : collecte capteurs

Flux principal :

- connexion d'un coach via Firebase
- detection des capteurs Bluetooth et USB
- demarrage d'un enregistrement au debranchement
- arret et export au rebranchement
- prediction automatique des sauts apres export

Cette application sert surtout a la prise de donnees. Pour le traitement detaille, la revue des sauts, l'annotation et le re-entrainement, il faut ensuite passer par `tools_gui.py`.

### `tools_gui.py` : guide detaille

L'outil est organise en onglets. Le flux normal va de `Data - New` vers `Data - Annotate`, puis vers les onglets de revue et enfin les onglets modeles.

#### Pour une premiere utilisation sans experience

Cette section donne le chemin le plus simple pour utiliser l'interface sans connaitre le code.

1. Ouvrir l'environnement Conda.
2. Lancer `tools_gui.py`.
3. Commencer par l'onglet `Start` pour relire le flux general.
4. Aller dans `Data`.
5. Cliquer sur le nom d'onglet `Data` pour ouvrir le menu des sous-ecrans, puis choisir `New`, `Annotate`, `Status`, etc.
6. Quand l'interface affiche un message dans la barre du bas, le lire avant de continuer : c'est souvent la meilleure indication de ce qui vient d'etre fait.

Commandes typiques :

```powershell
conda activate synergie-data
python tools_gui.py
```

Sous Windows, si l'environnement Conda est deja configure dans ce depot :

```powershell
& "C:\Users\micka\Documents\GIT\Synergie_Data\Synergie_Data\launch_tools_gui.bat"
```

Principe general :

- `Data` sert a preparer et annoter les donnees.
- `Review` sert a verifier les donnees et detecter les erreurs.
- `Models` sert a entrainer, comparer et comprendre les modeles.
- `Notes` sert d'aide-memoire.

Dans `Data`, `Review` et `Models`, les sous-ecrans sont dans un petit menu : cliquer sur le nom de l'onglet principal pour choisir le sous-ecran. L'objectif est de garder plus de place pour les videos, les graphes et les listes.

#### Vue d'ensemble des onglets

- `Start - Workflow` : rappel textuel du pipeline conseille de bout en bout
- `Data - Status` : inventaire des fichiers presents dans `data/new`, `data/pending`, `data/raw` et du volume de labels exploitables
- `Data - Sessions` : consultation et correction manuelle du registre de sessions dans `config/sessions.json`
- `Data - New` : point d'entree normal pour transformer une nouvelle seance brute en materiel d'annotation
- `Data - Process` : retraitement manuel ou batch de CSV IMU deja classes dans une session
- `Data - Annotate` : revue des sauts proposes, synchronisation video, correction des labels et finalisation vers le dataset
- `Review - Quality` : detection d'outliers et de sauts suspects dans le dataset annote
- `Review - Dataset` : revue d'exemples deja finalises dans `jumplist.csv`, avec exclusion rapide des cas douteux
- `Review - Turns` : audit de l'estimation du nombre de tours et comparaison annotation vs mesure IMU
- `Review - Inspect IMU` : visualisation d'un CSV brut, detection des sauts et ajustement des seuils
- `Review - Detection` : revue des faux positifs / faux negatifs deja identifies et balayage de seuils
- `Models - Train` : entrainement supervise des modeles `type` et `success`
- `Models - Importance` : importance de signaux par permutation pour comprendre ce que le modele utilise
- `Models - Tune` : recherche exploratoire d'hyperparametres
- `Models - Windows` : benchmark de tailles de fenetres et offsets de segment
- `Models - Audit` : synthese des modeles disponibles et de leur compatibilite
- `Notes` : aide-memoire sur l'algorithme et ses limites

#### Workflow recommande pour ajouter de nouvelles donnees

1. `Data - New`
2. `Data - Sessions`
3. `Data - Annotate`
4. `Review - Quality`
5. `Review - Detection`
6. `Models - Train`
7. `Models - Audit`, `Models - Importance`, `Models - Tune`

En pratique, le detail utile est le suivant :

1. Copier la nouvelle seance brute dans `data/new`.
2. Ouvrir `Data - New`, choisir le dossier detecte dans `Folder`, puis verifier les CSV listes dans `IMU files`.
3. Regarder la suggestion `Automatic session` et le chemin `Automatic annotation CSV`.
4. Cliquer `Add session automatically` si la session n'existe pas encore.
5. Cliquer `Process for annotation`.
6. Le traitement cree un fichier `*_for_annotation.csv` dans `data/pending`, des segments IMU, et tente de pre-remplir les colonnes d'annotation avec les modeles actuellement selectionnes.
7. Ouvrir `Data - Annotate`, charger ce fichier, revoir chaque saut, synchroniser la video et corriger les labels.
8. Quand tout est propre, lancer `Finalize annotated file`.
9. Utiliser `Review - Quality`, `Review - Dataset` et `Review - Detection` pour filtrer les exemples douteux avant re-entrainement.
10. Re-entrainer dans `Models - Train`.

Version tres concrete pour une nouvelle seance :

1. Copier le dossier de la seance dans `data/new`.
2. Ouvrir `tools_gui.py`.
3. Aller dans `Data`, puis choisir `New`.
4. Dans `Folder`, selectionner le dossier de la nouvelle seance.
5. Verifier que les CSV IMU apparaissent dans `IMU files`.
6. Cliquer `Add session automatically` si la session n'existe pas encore.
7. Cliquer `Process for annotation`.
8. Attendre que le log confirme la creation du fichier `*_for_annotation.csv`.
9. Aller dans `Data`, puis choisir `Annotate`.
10. Cliquer `Refresh annotation files`.
11. Selectionner le fichier de la seance.
12. Charger la video de la seance avec `Choose video...`.
13. Revoir chaque candidat de saut dans `Session Timeline`.
14. Corriger les informations de saut, succes, statut de revue et nombre de tours.
15. Sauvegarder regulierement avec `Save current annotation`, `Ctrl+S`, ou `Auto save`.
16. Quand toute la seance est terminee, cliquer `Finalize annotated file`.
17. Aller dans `Review - Quality` et `Review - Dataset` pour verifier les cas suspects.
18. Aller dans `Models - Train` seulement quand les annotations sont suffisamment propres.

#### `Data - Status`

Cet onglet sert de tableau de bord.

- Il aide a voir si de nouveaux CSV bruts attendent dans `data/new`.
- Il montre ce qui est deja passe en `pending`, ce qui est range en `raw`, et combien de labels sont reellement entrainables.
- Il est utile avant un training pour verifier qu'on travaille sur le bon volume de donnees.

#### `Data - Sessions`

Cet onglet est surtout un outil de maintenance.

- Le flux normal cree les sessions automatiquement depuis `Data - New`.
- Utiliser `Data - Sessions` surtout pour corriger une ancienne session, un mauvais chemin, ou une synchronisation manquante.
- Les champs de session pilotent notamment le dossier associe et l'offset `sample_time_fine_synchro` utilise ailleurs dans le GUI.

#### `Data - New`

C'est le point de depart prefere pour une nouvelle seance.

- `Folder` : liste les dossiers trouves dans `data/new`.
- `IMU files` : affiche les CSV detectes dans le dossier choisi.
- `Automatic annotation CSV` : montre le nom de sortie qui sera cree pour l'annotation.
- `Automatic session` : propose un identifiant et un chemin de session d'apres la date/heure du fichier.
- `Add session automatically` : ecrit cette session dans `config/sessions.json`.
- `Process for annotation` : lance la generation des segments et du CSV d'annotation.
- Le panneau de log en bas donne le detail des operations realisees.

Quand utiliser cet onglet :

- pour toute nouvelle seance jamais encore rangee
- pour creer proprement la session sans avoir a modifier `config/sessions.json` a la main
- pour preparer le materiel d'annotation sans passer par la CLI

#### `Data - Process`

Cet onglet sert a retraiter des CSV IMU deja ranges dans une session.

- `Session` : choisit une session connue.
- `Session CSV files` : liste les CSV bruts de cette session.
- `Batch process selected` : traite uniquement la selection.
- `Batch process all` : traite en masse tous les fichiers disponibles dans toutes les sessions.
- `Initial model predictions` : permet de choisir quel modele `type` et quel modele `success` utiliser pour generer les propositions initiales.

Cet onglet est utile si :

- tu veux recalculer des sorties apres un changement de modele
- tu veux retraiter un lot deja range dans `data/raw`
- tu veux reprendre seulement certains CSV sans refaire tout `data/new`

Ce n'est pas l'onglet recommande pour une session toute neuve si le flux `Data - New` suffit.

#### `Data - Annotate`

C'est l'onglet central pour construire un dataset propre.

Objectif de cet onglet :

- prendre une liste de candidats de saut detectes automatiquement
- regarder la video et le signal IMU
- confirmer le vrai type de saut
- corriger le nombre de tours et le succes
- marquer les signaux douteux pour qu'ils ne polluent pas l'entrainement
- ajouter les sauts manques si les capteurs ne les ont pas detectes

Avant de commencer :

- Avoir un fichier `*_for_annotation.csv` dans `data/pending`.
- Avoir la video de la seance accessible, idealement sur le disque local ou un disque externe branche.
- Savoir que la finalisation ne doit etre faite qu'a la fin de la revue de toute la seance.

Sur la gauche :

- `Refresh annotation files` recharge les `*_for_annotation.csv`
- la liste des fichiers montre ceux qui restent a traiter
- `Session Timeline` contient tous les candidats de saut de la seance, tries selon le temps video synchronise

Sur la droite :

- `Video Review` charge et relit une video de la seance
- `Choose video...` ouvre une popup pour choisir un dossier video, chercher les meilleurs matchs temporels et charger la bonne video
- `Clear cache (...)` vide les copies temporaires de videos creees pour accelerer la lecture depuis un disque externe
- `ADD JUMP` ouvre une popup pour chercher un saut manque sur un capteur avec des seuils de detection ajustables
- le systeme essaie d'utiliser les metadonnees video, le nom du fichier et les timestamps disque pour faire le rapprochement
- un offset de synchronisation par capteur peut etre memorise pour aligner IMU et video
- la timeline video, le slider et les boutons permettent de se deplacer rapidement

Procedure conseillee pour annoter un saut :

1. Selectionner un fichier dans la liste `Refresh annotation files`.
2. Selectionner le premier saut dans `Session Timeline`.
3. Charger ou verifier la video dans `Video Review`.
4. Cliquer le bouton pour aller au saut courant si la video n'est pas deja au bon endroit.
5. Regarder le mouvement dans la video.
6. Regarder le graphe `Jump Signals`.
7. Choisir le type de saut : piques, de carre, ou educatif.
8. Choisir le nombre de tours.
9. Choisir `Fall`, `Success` ou `Unknown`.
10. Choisir un `Review status`.
11. Cliquer `Save current annotation`, ou activer `Auto save`.
12. Passer au saut suivant.

Comment lire `Jump Signals` :

- La courbe bleue represente la vitesse angulaire principale du capteur.
- La courbe orange represente l'acceleration `Acc_X`.
- La courbe rouge represente `Gyr_X_ddot`, une derivee utilisee par la detection. Elle est affichee avec un facteur d'echelle pour etre visible.
- Les zones rouges transparentes montrent les moments ou le signal franchit le seuil de detection.
- Les traits verticaux indiquent les bornes du saut, notamment le takeoff et le landing quand ils sont disponibles.
- Le texte en haut explique pourquoi l'algorithme a considere ce segment comme un saut.
- Si le texte rouge dit que le timing est biomecaniquement suspect, il faut regarder la video et probablement marquer le cas comme `Weird signal / bad bounds` ou `No jump or drill`.

Important : un signal peut etre detecte comme un saut meme si la video montre autre chose. L'algorithme regarde des seuils de mouvement, pas l'intention du patineur. C'est exactement pour cela que la revue humaine existe.

Les controles video disponibles sont :

- `⏪` : reculer d'environ 1 seconde
- `◀` : reculer d'une frame
- `⏩` : avancer d'environ 1 seconde
- `▶` a droite du bouton lecture : avancer d'une frame
- `▶` / `⏸` : lecture / pause
- `Speed` : vitesse de lecture `1x`, `1.5x`, `2x` ou `3x`
- `⌖` : aller directement au saut courant
- `⏩J` : lecture acceleree jusqu'au saut courant
- `⏹` : stop

Raccourcis d'annotation visibles dans l'interface :

- `t / f / z / s / a` : type de saut
- `1 / 2 / 3 / 4` : nombre de tours
- `c / r / n` : chute / reussi / inconnu
- `u` : saut non visible sur la video
- `x` : signal bizarre ou bornes debut/fin incoherentes
- `l` : afficher ou masquer la legende du graphe
- `Espace` : lecture / pause video
- `Fleche gauche / droite` : reculer / avancer d'une frame
- `Ctrl+S` : sauvegarder l'annotation courante

Statuts de revue recommandes :

- `Seen jump` : le saut est visible et les bornes semblent coherentes.
- `Unseen on video` : la video ne montre pas le saut, ou la synchronisation ne permet pas de conclure.
- `Weird signal / bad bounds` : le signal est bizarre, les bornes takeoff/landing ne font pas de sens, ou le graphe ne ressemble pas a un saut utilisable.
- `Jump drill` : c'est un educatif de saut, pas un saut complet normal.
- `No jump or drill` : ce n'est pas un saut et ce n'est pas un educatif.

Choisir le type de saut :

- `Toe loop`, `Flip`, `Lutz` sont des sauts piques.
- `Salchow`, `Loop`, `Axel` sont des sauts de carre.
- Le nombre de tours se choisit dans la colonne `Turns`.
- Si le type ou le nombre de tours est incertain, corriger ce qui est visible et garder `Unknown` pour le succes si necessaire.

Sauvegarde :

- `Save current annotation` sauvegarde seulement le saut courant.
- `Ctrl+S` fait la meme chose rapidement au clavier.
- `Auto save` sauvegarde automatiquement apres chaque changement.
- `Finalize annotated file` doit etre utilise seulement quand toute la liste est revue. Cette action ajoute les annotations finalisees au dataset d'entrainement.

Synchronisation video :

- Le systeme essaie de trouver automatiquement une video proche de l'heure de la seance.
- Si la bonne video n'est pas chargee, cliquer `Choose video...`, choisir le dossier video, puis charger le meilleur match.
- Methode recommandee : aller au moment ou le tap sur un capteur est visible dans la video, selectionner la ligne `SYNC` correspondante dans `Session Timeline`, puis cliquer `Sync block impact at this frame`.
- Comme les capteurs sont lances ensemble par l'application, cette synchro est appliquee a tout le block de capteurs du fichier d'annotation.
- Methode de secours : si le tap n'est pas utilisable, placer la video exactement sur le saut selectionne et cliquer `Sync selected jump to block`.
- L'offset de block et sa source (`block_impact` ou `block_jump`) sont sauvegardes dans le fichier JSON `*.annotation_meta.json`, puis relus automatiquement quand on rouvre le fichier d'annotation.
- Quand une video est chargee, selectionner un saut dans `Session Timeline` deplace automatiquement la video au bon timecode calcule avec l'offset de block.
- La ligne de statut sous la video indique si la synchro de block est active et comment elle a ete faite.
- `Clear block sync` supprime la synchronisation sauvegardee pour le block courant.

Cache video local :

- Si une video est lue depuis un autre disque, par exemple `D:` sous Windows ou `/Volumes/...` sur Mac, le GUI peut la copier temporairement dans `.tmp/video_cache`.
- Si `ffmpeg` est disponible, le GUI peut aussi creer une proxy MP4 plus legere dans `.tmp/video_cache` pour rendre la lecture beaucoup plus fluide.
- Cette copie rend la lecture plus fluide et evite de relire constamment le disque externe.
- Le bouton `Clear cache (...)` montre la taille actuelle du cache.
- Cliquer `Clear cache (...)` libere l'espace disque utilise par ces copies temporaires.
- Vider le cache ne supprime pas la video originale.

Ajouter un saut manque :

- cliquer `ADD JUMP`
- choisir le capteur
- ajuster `Threshold`, `Sigma` et `Gap`
- cliquer `Detect candidates`
- selectionner le candidat voulu
- cliquer `Add selected jump`

La ligne ajoutee garde les reglages utilises dans les colonnes `detection_threshold`, `smoothing_sigma` et `combination_gap_frames`.

Quand ajouter un saut manuellement :

- La video montre clairement un saut, mais aucun candidat correspondant n'existe dans `Session Timeline`.
- Le saut est visible sur un autre capteur que celui actuellement selectionne.
- Les seuils automatiques ont manque un saut parce que le signal est plus faible, plus lent ou plus bruite que les autres.

Quand ne pas ajouter un saut :

- La video montre seulement une preparation, une transition ou un educatif sans saut complet.
- Le signal est tellement bruite qu'il n'est pas possible de definir un takeoff et un landing raisonnables.
- Le saut est deja present dans la liste avec un autre capteur et il suffit de corriger son label.

Conseils pratiques :

- utiliser `u` quand la video ne permet pas de conclure
- utiliser `x` quand le segment IMU parait mauvais, que les traits de debut/fin ne font pas de sens, ou qu'on ne veut pas reutiliser cet exemple pour l'entrainement
- utiliser `Jump drill` pour un educatif de saut capte comme un saut
- utiliser `No jump or drill` quand le candidat n'est ni un saut ni un educatif
- cocher `Auto save` pour sauvegarder automatiquement le saut courant apres chaque changement
- penser a `Finalize annotated file` seulement quand toute la seance est revue
- la finalisation est l'etape qui fusionne reellement les labels dans `data/annotated/total/jumplist.csv`

#### `Review - Quality`

Cet onglet sert a reperer des exemples suspects avant d'entrainer.

- `Run quality scan` inspecte le dataset annote
- la liste `Suspicious jumps to review` propose les cas atypiques
- les graphiques montrent les distributions et les outliers
- le panneau de details aide a comprendre pourquoi un saut est mis en evidence

Cas d'usage typiques :

- verifier des valeurs de rotation, duree ou acceleration anormales
- trouver des labels manifestement incoherents
- decider quels essais repasser en annotation ou exclure

#### `Review - Dataset`

Cet onglet permet de revoir des sauts deja finalises dans `data/annotated/total/jumplist.csv`.

- `Load trainable jumps` charge les exemples encore utilisables pour l'entrainement
- la liste affiche les sauts entrainables
- le panneau de droite montre les details et le signal du saut selectionne
- `Exclude selected (x)` retire rapidement un exemple du futur training

Le raccourci principal est :

- `x` : marque la ligne selectionnee comme `weird_signal`, la rend non entrainable et affiche une confirmation visuelle en rouge

Cet onglet est tres utile quand :

- un essai ancien parait mauvais apres coup
- tu veux nettoyer `jumplist.csv` sans reouvrir toute la session d'annotation
- tu veux retirer quelques cas clairement aberrants avant un nouveau training

#### `Review - Turns`

Cet onglet audite l'estimation du nombre de tours.

- `Run turn audit` compare les tours annotés avec la rotation mesuree par l'IMU
- la liste `Suspicious estimates` remonte les cas ou l'estimation fixe et l'annotation divergent
- le detail montre notamment les tours annotés, la rotation mesuree et l'estimation appliquee
- les graphiques aident a voir si l'erreur vient du signal ou de la regle d'estimation

Le systeme affiche la rotation cumulative en tours plutot qu'en degres, ce qui rend la lecture plus intuitive pour ce type d'audit.

#### `Review - Inspect IMU`

Cet onglet est l'outil visuel pour comprendre la detection de sauts sur un CSV brut.

- `Input CSV` : charge un CSV IMU
- `Session` : applique le contexte de session, notamment la synchronisation
- `Session CSV files` : permet de double-cliquer sur un fichier d'une session pour l'ouvrir vite
- `2nd derivative threshold` : seuil principal de detection
- `Smoothing sigma` : lissage du gyroscope
- `Combination gap (frames)` : ecart utilise pour identifier les combinaisons
- `Load and detect` : recharge le CSV et relance la detection
- `Refresh plots` : redessine les graphiques avec les parametres actuels

Le panneau `Detected Jumps` liste les sauts trouves, et la zone de graphiques montre les signaux relies a la detection. Cet onglet est le meilleur endroit pour comprendre pourquoi un faux positif ou un faux negatif s'est produit.

#### `Review - Detection`

Cet onglet s'appuie sur les erreurs deja revues pendant l'annotation.

- les lignes marquees `Not a jump` deviennent des faux positifs connus
- les lignes ajoutees manuellement deviennent des faux negatifs connus
- `Refresh reviewed errors` recharge cet ensemble d'exemples
- `Run threshold sweep` teste plusieurs seuils et niveaux de lissage pour proposer un meilleur compromis

Il faut idealement utiliser cet onglet apres plusieurs vraies revues d'annotation, sinon le jeu d'erreurs reste trop petit pour guider un reglage utile.

#### `Models - Train`

Cet onglet pilote l'entrainement principal.

Parametres importants :

- `Task` : `type` ou `success`
- `Architecture` : `inceptiontime` ou `transformer` pour `type`, `tcn` ou `lstm` pour `success`
- `Dataset` : dossier contenant `jumplist.csv` et `skaterData.csv`
- `Epochs`
- `Parameter profile` : parametres par defaut ou profil optimise issu de `Models - Tune`
- `Use weight and height` : active ou neutralise les variables scalaires athlète
- `Batch size`, `Learning rate`, `Dropout`, `Filters / units`, `Modules / blocks`
- `Start from pretrained model` : repart d'un modele compatible au lieu de zero

Sorties visibles :

- courbes d'entrainement
- log d'entrainement
- matrice de confusion
- resume dataset / modeles pre-entraînés / qualite

Bon usage :

- rafraichir d'abord les stats dataset
- nettoyer les cas douteux avec `Review - Quality` et `Review - Dataset`
- verifier ensuite le modele dans `Models - Audit`

Avant l'entrainement, les valeurs d'acceleration impossibles au-dela de `32 g` sont remplacees par interpolation dans les segments charges en cache. Cela evite qu'un spike aberrant domine le modele sans lisser les signaux normaux.

#### `Models - Importance`

Cet onglet sert a comprendre quels signaux sont vraiment utilises par un modele deja entraine.

- choisir `Task`
- selectionner un modele sauvegarde ou l'alias du modele actif
- regler `Repeats` et `Time windows`
- lancer `Run signal importance`

Le graphique de permutation aide a voir quelles composantes temporelles ou quelles fenetres influencent le plus la performance.

#### `Models - Tune`

Cet onglet lance une recherche exploratoire d'hyperparametres.

- `Task`, `Architecture`, `Dataset`
- `Trials`
- `Epochs / trial`
- option `Use weight and height`
- `Run search`

Le resultat montre :

- un classement des meilleurs essais
- une barre de progression
- un log detaille de la recherche

L'objectif n'est pas de remplacer l'entrainement courant a chaque fois, mais de comparer quelques candidats, puis de reporter le meilleur profil dans `Models - Train`.

#### `Models - Windows`

Cet onglet compare plusieurs fenetres temporelles et offsets de segment.

- `Frames` : tailles de fenetres a comparer
- `Epochs / window`
- `Offset window frames`
- `Offsets in segment`
- `Run window benchmark`
- `Run offset benchmark`

Il est utile quand on soupconne que le segment coupe trop tot ou trop tard autour du saut, ou qu'une autre longueur de sequence donnerait un meilleur compromis.

#### `Models - Audit`

Cet onglet donne une synthese sur les modeles disponibles et leur compatibilite.

- `Run model audit` produit un rapport texte
- le rapport aide a voir quels modeles sont presents, actifs, compatibles avec le code actuel, ou potentiellement a archiver

Avant de reutiliser un modele pour pre-remplir de nouvelles annotations, c'est un bon dernier controle.

#### `Notes`

Cet onglet contient un aide-memoire sur l'algo actuel, par exemple :

- la segmentation repose surtout sur les derivees de `Gyr_X`
- la rotation signee est conservee pour le diagnostic
- l'onglet `Review - Inspect IMU` est l'outil de base pour comprendre les seuils et les erreurs de detection

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

### Strategie espace disque

Les CSV texte prennent beaucoup de place, surtout parce que les memes echantillons IMU peuvent etre recopies dans plusieurs segments. La strategie recommandee est de garder les sessions brutes comme source canonique, de stocker les annotations avec les indices/temps takeoff-landing, puis de generer des fenetres compactes en cache pour l'entrainement.

Voir [docs/storage_strategy.md](C:\Users\micka\Documents\GIT\Synergie_Data\Synergie_Data\docs\storage_strategy.md) pour le plan de migration propose.

### Manifest de sauvegarde

Pour creer un fichier lisible qui regroupe les sessions, les essais en attente, les essais finalises et les metadonnees video/synchro :

```bash
python main.py export-backup-manifest --output data/backup_manifest.json
```

Le fichier JSON produit ne duplique pas les gros CSV/video. Il sert de carte de sauvegarde/audit avec les chemins, les statuts, les labels, les temps takeoff/landing et les metadonnees utiles pour comprendre ou reconstruire le dataset.

Pour creer en plus une archive HDF5 compacte derivee du manifest :

```bash
python main.py export-hdf5-archive --output data/synergie_archive.h5
```

L'approche hybride recommandee est de garder `backup_manifest.json` comme carte lisible et `synergie_archive.h5` comme conteneur numerique rapide. Le HDF5 embarque le manifest et les segments IMU compresses en `float32`; il peut etre regenere depuis les CSV et ne devrait pas etre versionne dans Git.

Pour verifier quels CSV de segments sont deja representes dans le HDF5 avant de les archiver ou supprimer manuellement :

```bash
python main.py plan-segment-cleanup --archive data/synergie_archive.h5
```

Cette commande ne supprime rien. Elle donne seulement le nombre de fichiers candidats et l'espace disque recuperable.

Pour archiver les CSV couverts tout en gardant une possibilite de retour arriere, utiliser d'abord le dry-run :

```bash
python main.py archive-segment-csvs --archive data/synergie_archive.h5 --destination data/segment_csv_archive
```

Puis, seulement apres validation :

```bash
python main.py archive-segment-csvs --archive data/synergie_archive.h5 --destination data/segment_csv_archive --apply
```

Cette commande deplace les CSV dans `data/segment_csv_archive` en conservant leur chemin relatif. Les lecteurs principaux essaient d'abord le HDF5 puis reviennent au CSV si necessaire.

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
