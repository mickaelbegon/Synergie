# Guide Maksim — labelling sûr et audit de détection

## Règle principale

Le travail déjà labellé est une source de référence. Ne jamais écraser, déplacer,
supprimer ou régénérer un fichier `*_for_annotation*.csv`, un segment associé ou
un fichier `*.annotation_meta.json` déjà travaillé. Les corrections de détection
seront évaluées sur des copies et ne modifieront pas ces fichiers.

## Avant chaque lot

1. Fermer l'écran d'annotation et vérifier que toutes les modifications sont
   enregistrées.
2. Créer le manifest de sauvegarde depuis la racine du projet :

   ```powershell
   python main.py export-backup-manifest --output data/backups/manifest_YYYYMMDD_HHMM.json
   ```

3. Copier le CSV d'annotation, son `*.annotation_meta.json` et les segments IMU
   correspondants dans un dossier daté de sauvegarde. Ne pas remplacer une
   sauvegarde existante.
4. Noter le chemin du manifest et de la copie dans le rapport de lot.

## Labelling et classification des anomalies

Utiliser les statuts habituels de l'outil pour confirmer un saut, indiquer un
non-saut, un signal problématique ou ajouter un saut manqué. Pour les cas utiles
à la correction du détecteur, ajouter dans une **copie dédiée à l'audit** une
colonne `detection_issue_category` et, si connu, `rotation_direction`.

Valeurs de `detection_issue_category` :

- `immobile`
- `slow_movement` (walkthrough ou mouvement trop lent)
- `twizzle`
- `flying_spin`
- `walley`
- `duplicate_detection`
- `left_rotation`
- `missed_jump`

Valeurs de `rotation_direction` : `left` ou `right`. Laisser vide si le sens est
inconnu. Un saut manqué doit aussi être ajouté avec le mécanisme normal de
l'outil afin de conserver son segment et son horodatage; la catégorie ne remplace
pas le label.

Ne pas modifier les catégories d'un fichier original déjà labellé uniquement
pour l'audit. Conserver plutôt une copie portant par exemple le suffixe
`_audit_copy.csv`; elle ne doit pas être finalisée ni utilisée pour entraîner un
modèle.

## Rapport d'audit sans modification

L'audit lit uniquement les CSV et peut produire un JSON traçable avec les
empreintes SHA-256 des fichiers source :

```powershell
python main.py audit-detection-regressions --root data/pending --output data/reports/detection_audit_YYYYMMDD.json
```

Sans `--output`, il affiche le résultat et n'écrit aucun fichier. Le rapport ne
change pas les labels, les métadonnées, les segments ou les paramètres du
détecteur.

## Rapport à transmettre après un lot

Transmettre :

- la liste des sessions et capteurs traités;
- le chemin du manifest de sauvegarde et de la copie de sauvegarde;
- les CSV d'annotation, métadonnées et segments associés (copies seulement);
- le rapport JSON d'audit;
- le nombre de vrais sauts, faux positifs et sauts manqués;
- le détail des catégories ci-dessus, le sens de rotation lorsqu'il est connu,
  et les cas ambigus;
- toute duplication observée, avec les noms complets de chaque fichier;
- la confirmation qu'aucun fichier déjà labellé n'a été écrasé, supprimé ou
  finalisé à nouveau.

En cas de doute sur une session ou un doublon, arrêter la finalisation de cette
session et la signaler dans le rapport. Garder les originaux inchangés permet de
revenir en arrière et de comparer toute future correction aux annotations
actuelles.
