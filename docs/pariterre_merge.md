# Analyse de la branche `pariterre`

Comparaison observée le 6 mai 2026 entre `main` (`28f7d05`) et `pariterre/main` (`6c7beec`).

## Ce que la branche apporte

- une migration vers un package `synergie/`
- une documentation plus fournie
- des fichiers d'environnement Conda
- des corrections de robustesse sur les connexions capteurs

## Pourquoi ne pas fusionner en bloc

- la branche est large : `22` commits d'avance et `1` commit de retard
- elle déplace ou renomme presque toute l'arborescence
- elle mélange refactor, packaging, UI et logique de connexion matérielle

## Recommandation

1. Stabiliser `main` avec des petites améliorations testables.
2. Reprendre ensuite les idées utiles de `pariterre` par lots séparés.
3. Réserver les gros déplacements de fichiers à une PR dédiée.
4. Revalider la GUI avec du matériel réel avant tout merge USB/Bluetooth plus profond.

## Changement à reprendre en priorité

Le commit `6c7beec` ("Added a try catch") ajoute une protection autour de la réouverture USB dans le gestionnaire des capteurs. C'est le correctif le plus simple à cherry-pick ensuite.
