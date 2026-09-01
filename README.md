# Pipeline hydrodynamique AUV — OpenFOAM

Ce dépôt transforme une coque AUV solide au format STEP en une analyse CFD reproductible : détection de l’étrave, normalisation de l’axe d’avance, trois maillages de qualification à 1,00 m/s, balayage sur le maillage moyen de 0,50 à 2,00 m/s, puis ajout transactionnel des résultats à une table cumulative.

Le pipeline n’exécute aucun test logiciel et n’ajoute aucune fonction de cybersécurité. Ses contrôles sont uniquement des validations opérationnelles de CAO, maillage, convergence numérique et physique CFD.

## Installation sur une autre machine

La configuration de référence utilise Fedora 42, Python 3 et OpenFOAM v2512. OpenFOAM publie officiellement des paquets v2512 pour Fedora 41/42 ; le dépôt COPR `openfoam/openfoam` fournit le paquet `openfoam2512-default` utilisé par ce pipeline.

```bash
git clone git@github.com:armeldemarsac92/simple_cfd.git
cd simple_cfd
sudo dnf copr enable openfoam/openfoam
sudo dnf install openfoam2512-default python3 python3-pip
./setup.sh
./analyze_hull.sh --doctor
```

La documentation officielle est disponible dans les pages [Installation on Linux](https://www.openfoam.com/download/openfoam-installation-on-linux) et [OpenFOAM v2512](https://www.openfoam.com/news/main-news/openfoam-v2512).

Si le fichier d’environnement v2512 se trouve ailleurs, indiquer son chemin sans modifier le dépôt :

```bash
OPENFOAM_BASHRC=/path/to/OpenFOAM-v2512/etc/bashrc ./analyze_hull.sh --doctor
OPENFOAM_BASHRC=/path/to/OpenFOAM-v2512/etc/bashrc ./analyze_hull.sh 'Part Studio 1 - Part 1.step'
```

La coque ayant servi à créer le pipeline est incluse sous le nom `Part Studio 1 - Part 1.step`. Une machine destinée au protocole par défaut doit disposer d’au moins 30 Gio de mémoire physique, 30 Gio de stockage libre et 8 cœurs physiques.

## Utilisation courante

Préparer l’environnement une fois :

```bash
./setup.sh
./analyze_hull.sh --doctor
```

Analyser directement un fichier, y compris si son chemin contient des espaces :

```bash
./analyze_hull.sh 'path/with spaces/design.step'
```

Ou déposer une ou plusieurs coques dans la boîte d’entrée :

```bash
cp 'new-design.step' hulls/inbox/
./analyze_hull.sh
```

Sans argument, tous les fichiers `hulls/inbox/*.step` et `*.stp` sont analysés dans l’ordre de leur nom. Chaque coque doit représenter un unique solide fermé, être exprimée dans une unité STEP prise en charge et mesurer entre 0,05 et 20 m dans sa plus grande dimension.

## Étrave et direction d’avance

Le pipeline détecte l’axe principal et choisit automatiquement l’étrave à partir de la géométrie des deux extrémités. Il refuse une décision dont la confiance est inférieure à 0,70. Pour une coque ambiguë, indiquer explicitement l’étrave dans les coordonnées du fichier source :

```bash
./analyze_hull.sh --bow=-y 'ambiguous-design.step'
```

La coque normalisée est orientée de sorte que l’écoulement relatif soit en `+X`. L’orientation originale, la matrice de transformation, son inverse et une prévisualisation sont conservés dans le run.

## Conditions physiques

La configuration par défaut représente :

- une coque fixe, entièrement immergée, au centre d’une profondeur de 5 m ;
- de l’eau de mer à 15 °C et 35 PSU, `rho = 1025 kg/m³`, `nu = 1,17e-6 m²/s` ;
- une incidence et un lacet nuls ;
- sept vitesses : 0,50, 0,75, 1,00, 1,25, 1,50, 1,75 et 2,00 m/s ;
- un calcul transitoire incompressible URANS SST k-omega, parois lisses et fonctions de paroi ;
- une détection automatique du transitoire initial, puis une moyenne temporelle cumulée arrêtée lorsque son incertitude de convergence ITTC et sa dérive deviennent inférieures à 0,5 %.

Trois grilles sont calculées à 1,00 m/s. Le balayage de production réutilise ensuite la grille moyenne. L’incertitude de discrétisation (ordre observé et GCI) n’est donc établie qu’à **1,00 m/s** et est étiquetée `reference_speed_only` aux autres vitesses.

Ce modèle exclut la surface libre, les vagues, les appendices absents de la CAO, la propulsion, les gouvernes braquées, les manœuvres, la rugosité, la transition laminaire-turbulente et la cavitation multiphasique. La marge de cavitation fournie est seulement un écran de pression statique à la profondeur configurée.

> This is a smooth, fully turbulent, fixed-hull URANS prediction and has not been validated against experiments.

## Déroulement et reprise

Une commande normale recherche un run compatible avec le SHA-256 de la CAO, la configuration et le build OpenFOAM. Elle réutilise chaque étape CFD déjà acceptée et reprend au prochain jalon. Les calculs ne sont jamais lancés en parallèle les uns avec les autres.

- `--reference-only` s’arrête après les trois calculs à 1,00 m/s et l’étude GCI.
- `--stop-after reference` fait le même arrêt dans le parcours complet ; relancer ensuite la commande reprend au balayage.
- `--restart` crée un nouveau run complet sans réutiliser le précédent.
- `--force` crée également un nouveau run numérique et, lors de la persistance, une nouvelle révision sans écraser les lignes antérieures.
- `--report-only RUN_ID` régénère les visualisations et rapports à partir des champs finaux d’un run existant, sans relancer le solveur.

Les artefacts d’anciennes versions du pipeline ne sont pas un contrat de compatibilité. Seul le schéma courant est autoritatif : après un changement de protocole, les runs et tables produits par l’ancien schéma doivent être supprimés puis recalculés par la commande normale. Il n’existe ni migration implicite, ni branche de compatibilité, ni réglage propre à une coque.

## Résultats

Chaque exécution possède un dossier immuable :

```text
runs/<horodatage>-<sha-cao>-<sha-config>/
├── manifest.json
├── status.json
├── run-events.jsonl
├── geometry/
├── meshes/
├── cases/
├── postprocessing/grid-study.json
└── reports/
    ├── report.md
    ├── report.html
    └── figures/
```

Les résultats cumulés, mis à jour seulement après les décisions numériques, sont :

- `results/results.sqlite` : données complètes, y compris les grilles grossière et fine et les diagnostics ;
- `results/summary.csv` : une ligne acceptée par coque et vitesse de production, prête pour un tableur ;
- `results/comparison.html` : comparaison automatique de toutes les coques acceptées.

Chaque cas accepté conserve les forces totale/pression/visqueuse, les moments, `Cd`, la puissance de remorquage idéale, Reynolds, résidus, fenêtre de convergence, statistiques `y+`, `Cp_min`, pression absolue minimale et marge de cavitation. Son dossier `visualizations/` contient :

- la carte `Cp` sur la coque ;
- la contrainte de cisaillement pariétal ;
- la carte `y+` ;
- une coupe longitudinale de vitesse et pression avec lignes de courant ;
- les VTK bruts et un fichier `case.foam` ouvrable dans ParaView.

## Coût de calcul

La configuration utilise 8 rangs MPI et exige au moins 30 Gio de RAM physique et 30 Gio libres avant le départ. Sur la coque de référence, les deux premiers maillages du protocole courant contiennent 525 433 et 1 220 794 cellules ; le compte fin est consigné par le run de qualification. Une analyse complète comporte neuf solveurs distincts : trois grilles à 1,00 m/s, puis six vitesses supplémentaires sur la grille moyenne. Leur durée n’est pas fixée par un nombre arbitraire d’itérations : chaque solveur s’arrête au premier jalon dont la moyenne de traînée satisfait les critères statistiques, avec un plafond de 20 temps de traversée. Prévoir plusieurs heures ; les temps et consommations réels de chaque commande sont enregistrés dans les artefacts du run.

Les partitions MPI temporaires sont supprimées après reconstruction et post-traitement réussis. Le maillage accepté, le dernier champ reconstruit, les historiques, les journaux, les VTK et les rapports restent disponibles.
