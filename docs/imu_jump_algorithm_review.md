# Audit de l'algorithme IMU

## Pipeline actuel

1. Pre-traitement dans `trainingSession` a partir de `Gyr_X`.
2. Detection des bornes de saut via la derivee seconde lissee de `Gyr_X`.
3. Extraction d'une fenetre temporelle autour du saut.
4. Calcul de la rotation par integration de `Gyr_X`.
5. Prediction du type et du succes via deux modeles.

## Observation sur le sens de rotation

Le code actuel a bien un risque de restriction implicite sur le sens de rotation :

- la detection de saut part uniquement de `Gyr_X`
- la rotation exportee et utilisee en aval est prise en valeur absolue
- si le dataset d'entrainement est majoritairement dans un seul sens, le modele peut apprendre ce biais via les series temporelles brutes

La modification recente conserve maintenant aussi :

- `signed_rotations`
- `rotation_direction` (`positive`, `negative`, `unknown`)

Ces champs sont diagnostiques. Ils permettent d'auditer si les predictions changent selon le signe de rotation.

## Recommandations

- verifier la repartition des sens de rotation dans le dataset annote
- tester un miroir de donnees IMU en inversant les axes de rotation pertinents
- rendre la detection moins dependante du signe brut de `Gyr_X`
- documenter precisement la convention de signe du capteur
