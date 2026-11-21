# Démarche du projet --- Segmentation d'images de télédétection

## 1. Présentation générale

Ce projet consiste à développer une solution de segmentation sémantique
d'images aériennes de télédétection. L'objectif est d'attribuer une
classe à chaque pixel de l'image afin de produire une carte thématique
de l'occupation du sol.

La solution repose sur une architecture **U-Net**, entraînée sur le jeu
de données **LandCover.ai**, puis intégrée dans un backend Python qui
pourra être utilisé par une interface Streamlit.

### Classes reconnues

Le modèle prédit cinq classes :

    ID Classe       Couleur utilisée
  ---- ------------ ------------------
     0 Background   Noir
     1 Building     Rouge
     2 Woodland     Vert
     3 Water        Bleu
     4 Road         Jaune

**Limite importante du jeu de données :** LandCover.ai fournit des
annotations pour les bâtiments, les zones boisées, l'eau et les routes,
avec une classe de fond. Il ne fournit pas de classes dédiées à
l'agriculture ou au sol nu. Le modèle ne doit donc pas être présenté
comme capable de distinguer explicitement ces deux catégories.

------------------------------------------------------------------------

## 2. Démarche suivie

Le travail a été réalisé en plusieurs étapes.

### Étape 1 --- Étude et préparation du jeu de données

Le jeu de données utilisé est **LandCover.ai**. Il contient des
orthophotos aériennes et leurs masques de segmentation associés.

Les premières vérifications ont porté sur :

-   le nombre d'images et de masques ;
-   la correspondance entre chaque image et son masque ;
-   les dimensions des images ;
-   le format des données ;
-   les valeurs de pixels présentes dans les masques ;
-   la distribution des classes.

Les données ont été organisées en ensembles d'entraînement, de
validation et de test, selon les partitions préparées pour le projet.

Les grandes images ont été découpées en patches de **512 × 512 pixels**,
afin de faciliter l'entraînement du réseau.

### Étape 2 --- Prétraitement

Les images sont des images RGB. Pour l'entraînement, elles sont
converties en tenseurs et normalisées en divisant les valeurs des pixels
par 255.

Les masques sont conservés sous forme d'indices entiers de classes, de 0
à 4. Ils ne sont pas traités comme des images RGB : chaque pixel
contient directement l'identifiant de sa classe.

Les dimensions et types attendus par le modèle sont :

-   Images : `[batch, 3, 512, 512]`, de type `float32`.
-   Masques : `[batch, 512, 512]`, de type `int64`.

### Étape 3 --- Architecture du modèle

L'architecture retenue est **U-Net**, un réseau de neurones
convolutionnel souvent utilisé pour la segmentation d'images.

Elle comprend :

1.  Un chemin encodeur qui extrait progressivement les caractéristiques
    de l'image.
2.  Un bloc central (*bottleneck*) qui apprend des représentations plus
    profondes.
3.  Un chemin décodeur qui reconstruit la carte de segmentation.
4.  Des connexions de saut (*skip connections*) qui transmettent au
    décodeur des informations spatiales issues de l'encodeur.

Le modèle reçoit une image RGB de taille 512 × 512 et produit des scores
pour les cinq classes à chaque pixel.

La fonction de perte utilisée est `CrossEntropyLoss`. Elle compare les
scores prédits par le réseau avec les indices de classes du masque réel.

### Étape 4 --- Entraînement

L'entraînement a été effectué sur Kaggle avec un GPU Tesla T4, afin
d'accélérer les calculs.

Le modèle a été entraîné pendant 30 époques. À chaque époque, les
performances ont été contrôlées sur l'ensemble de validation.

Le checkpoint retenu est **`unet_best.pth`**, sauvegardé à la meilleure
époque de validation, soit l'époque 19. Il contient notamment les poids
du modèle et les informations de l'époque et de la perte de validation.

Le checkpoint de meilleure validation est utilisé pour l'inférence,
plutôt que de sélectionner automatiquement la dernière époque.

### Étape 5 --- Évaluation

Le modèle a été évalué sur l'ensemble de test, qui contient 1 602
patches.

Les métriques obtenues sont :

  Métrique           Résultat
  ---------------- ----------
  Pixel Accuracy      88,53 %
  Mean IoU            50,10 %
  Mean Dice           61,42 %

Résultats par classe :

  Classe             IoU      Dice
  ------------ --------- ---------
  Background     83,03 %   90,73 %
  Building       26,74 %   42,19 %
  Woodland       83,72 %   91,14 %
  Water          45,67 %   62,70 %
  Road           11,32 %   20,34 %

L'accuracy globale est élevée, mais elle doit être interprétée avec
prudence en raison du déséquilibre entre les classes. Les classes
Background et Woodland sont beaucoup plus représentées que Building et
Road. Les résultats montrent que la détection des routes, et dans une
moindre mesure des bâtiments, reste difficile.

### Étape 6 --- Développement du module d'inférence

Le module `src/inference/predict.py` permet de réutiliser le modèle
entraîné sans relancer l'entraînement.

Il expose principalement deux fonctions :

-   `load_model()` : charge l'architecture U-Net et le checkpoint.
-   `predict_image(image, model, device)` : effectue la segmentation et
    retourne les résultats.

Pour les grandes images, l'inférence est effectuée par tuiles de 512 ×
512 pixels. Chaque tuile est prédite séparément, puis les résultats sont
replacés à leur position initiale pour reconstruire le masque à la
taille de l'image.

Les zones de bord qui ne mesurent pas exactement 512 pixels sont
complétées temporairement avant la prédiction, puis seule la partie
correspondant à l'image est conservée.

La fonction retourne un dictionnaire contenant :

  -----------------------------------------------------------------------
  Clé                                 Contenu
  ----------------------------------- -----------------------------------
  `original`                          Image RGB originale sous forme de
                                      tableau NumPy

  `mask`                              Masque prédit contenant les IDs de
                                      classes

  `colored_mask`                      Masque coloré pour la visualisation

  `overlay`                           Superposition de l'image et du
                                      masque coloré

  `class_percentages`                 Pourcentage de pixels prédits dans
                                      chaque classe
  -----------------------------------------------------------------------

Les tests réalisés ont confirmé le chargement du checkpoint, l'inférence
sur des images PNG et JPG, ainsi que la production de masques et
d'overlays. Les tests d'inférence par tuiles ont également été exécutés
sur des images aériennes.

**À connaître :** la version actuelle utilise des tuiles sans
chevauchement. Des discontinuités peuvent donc apparaître aux frontières
entre deux tuiles. De plus, la conversion en RGB pour la prédiction ne
préserve pas les métadonnées géographiques des GeoTIFF. Une sortie
géoréférencée nécessiterait un traitement spécifique avec Rasterio.

------------------------------------------------------------------------

## 3. Organisation du projet

La structure générale du dépôt est organisée par responsabilité. Les
noms exacts de certains dossiers peuvent varier selon la version finale
du dépôt.

``` text
segmentation_landcover/
│
├── app/                         # Interface utilisateur Streamlit
│
├── checkpoints/
│   └── unet_best.pth            # Meilleurs poids entraînés
│
├── configs/                     # Fichiers de configuration
│
├── data/
│   ├── raw/                     # Données originales
│   ├── processed/               # Données prétraitées et patches
│   └── splits/                  # Fichiers de partition, si utilisés
│
├── docs/                        # Documentation du projet
│
├── notebooks/                  # Notebooks d’exploration et d’expérimentation
│
├── reports/                     # Visualisations et résultats
│
├── src/
│   ├── dataset/                 # Chargement et préparation des données
│   ├── models/
│   │   └── unet.py              # Architecture U-Net
│   ├── training/                # Entraînement et évaluation
│   └── inference/
│       ├── __init__.py
│       ├── predict.py           # API d’inférence
│       ├── test_real_image.py   # Test sur une image
│       └── test_formats.py      # Test JPG/PNG
│
├── requirements.txt
├── README.md
└── .gitignore
```

Les dossiers `data/` et `notebooks/` peuvent être volumineux. Avant le
transfert, il faut vérifier que les données sont autorisées à être
redistribuées et que le mode de partage choisi convient à la licence du
dataset. Le checkpoint et le code d'inférence sont nécessaires pour
exécuter le modèle déjà entraîné ; les données complètes ne sont pas
nécessaires pour une simple prédiction.

------------------------------------------------------------------------

## 4. Installation et exécution

### Prérequis

-   Python 3.10 ou une version compatible avec les dépendances du
    projet.
-   Un environnement virtuel Python recommandé.
-   PyTorch installé avec une version adaptée à la machine (CPU ou GPU).

### Créer l'environnement

``` bash
python -m venv .venv
source .venv/bin/activate
```

### Installer les dépendances

``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

Pour PyTorch, il peut être nécessaire de choisir la commande
d'installation correspondant à l'environnement CPU ou CUDA. Se référer à
la page officielle d'installation de PyTorch si l'installation générique
de `requirements.txt` ne correspond pas à la machine.

### Tester le chargement et l'inférence

Depuis la racine du projet :

``` bash
python -m src.inference.test_real_image
```

Le test de formats peut être lancé avec :

``` bash
python -m src.inference.test_formats
```

Ces scripts de test peuvent dépendre des chemins d'images présents dans
le dépôt. Vérifier leurs chemins d'entrée avant de les exécuter sur une
autre machine.

------------------------------------------------------------------------

## 5. Intégration avec Streamlit

L'interface peut importer directement les fonctions du backend. Le
modèle doit être chargé une seule fois au démarrage ou mis en cache côté
Streamlit, et non rechargé à chaque prédiction.

Exemple d'utilisation :

``` python
from src.inference.predict import load_model, predict_image

model, device = load_model()

# image est une image PIL ouverte par Streamlit
result = predict_image(image, model, device)

original = result["original"]
mask = result["mask"]
colored_mask = result["colored_mask"]
overlay = result["overlay"]
percentages = result["class_percentages"]
```

L'interface peut ensuite afficher :

-   l'image originale ;
-   la carte de segmentation colorée ;
-   l'overlay ;
-   la répartition en pourcentage des classes.

Il est recommandé d'afficher clairement le nom des classes et leur
couleur, ainsi qu'un avertissement indiquant que les résultats sont des
prédictions automatiques.

------------------------------------------------------------------------

## 6. Limites et améliorations possibles

Les points suivants doivent être connus lors de la présentation et de
l'utilisation du projet :

1.  **Déséquilibre des classes :** les classes peu représentées, en
    particulier Road, sont moins bien détectées.
2.  **Limites des annotations :** le modèle reconnaît les cinq
    catégories du dataset et ne distingue pas explicitement
    l'agriculture ou le sol nu.
3.  **Frontières des tuiles :** l'inférence sans chevauchement peut
    produire des ruptures visibles entre patches.
4.  **Géoréférencement :** le résultat actuel est une image de
    segmentation, pas encore un raster de sortie conservant les
    métadonnées géographiques du GeoTIFF.
5.  **Généralisation :** les performances mesurées sur LandCover.ai ne
    garantissent pas les mêmes résultats sur des images provenant
    d'autres régions, capteurs ou conditions.
6.  **Prétraitement :** les images utilisées en production doivent être
    compatibles avec les images RGB et le prétraitement employés pendant
    l'entraînement.

Améliorations envisageables : tuiles avec chevauchement, fusion des
prédictions dans les zones communes, prise en charge GeoTIFF avec
conservation du géoréférencement, enrichissement des données et
amélioration de la détection des classes minoritaires.

------------------------------------------------------------------------

## 7. Répartition des responsabilités pour la suite

### Backend / modèle

Le backend fournit :

-   le modèle U-Net et son checkpoint ;
-   le chargement du modèle ;
-   la fonction de prédiction ;
-   le masque, l'overlay et les pourcentages ;
-   la documentation des classes, des métriques et des limites.

### Frontend / Streamlit

L'interface doit :

-   permettre de sélectionner ou téléverser une image ;
-   appeler `load_model()` une seule fois ;
-   appeler `predict_image()` lors de la demande de segmentation ;
-   afficher l'image, le masque et l'overlay ;
-   présenter les pourcentages par classe ;
-   gérer les erreurs de format, de taille et de chargement.

### Intégration finale

Avant la démonstration, effectuer un test complet depuis l'interface :
téléverser une image, lancer la prédiction, afficher les résultats et
vérifier que les dimensions sont cohérentes.

------------------------------------------------------------------------

## 8. Résumé

Le projet suit le pipeline suivant :

**LandCover.ai → vérification et préparation des données → patches 512 ×
512 → entraînement U-Net sur Kaggle → sélection du meilleur checkpoint →
évaluation sur le jeu de test → inférence par tuiles → affichage des
résultats dans Streamlit.**

Le modèle retenu est le checkpoint `unet_best.pth`, sauvegardé à
l'époque 19. Les métriques d'évaluation sont de 88,53 % en Pixel
Accuracy, 50,10 % en Mean IoU et 61,42 % en Mean Dice.

Cette documentation décrit la démarche et le fonctionnement de la
version actuelle. Les limites mentionnées doivent être conservées dans
la présentation afin de distinguer les fonctionnalités réellement
implémentées des améliorations futures.
