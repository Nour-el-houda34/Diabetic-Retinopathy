📘 Classification de la Rétinopathie Diabétique (Deep Learning + Interface Web)

Projet complet de Deep Learning ayant pour objectif la classification automatique des images de fond d’œil en 5 niveaux de sévérité de la Rétinopathie Diabétique, intégrant :

🧠 Un modèle de Deep Learning TensorFlow/Keras

🌐 Une interface Web (Flask)

📊 Un pipeline complet : data exploration → entraînement → évaluation → déploiement

🖼️ Un système d’upload + prédiction + historique

Projet réalisé dans le cadre du module :
“Projet Deep Learning — ISI M2 — Année 2025–2026”

🎯 Objectif du Projet

Développer un système d’aide au diagnostic capable de classifier automatiquement les images de rétinopathie diabétique en 5 classes :

Label	Classe
0	Healthy (Sain)
1	Mild DR (Léger)
2	Moderate DR (Modéré)
3	Severe DR (Sévère)
4	Proliferative DR (Proliférative)

Le modèle analyse l’image de fond d’œil et renvoie la classe correspondante avec un score de probabilité.

🧰 Technologies Utilisées
Backend & Deep Learning

Python 3.8+

TensorFlow 2.x / Keras

Scikit-learn

Pandas / NumPy

Traitement d’images

OpenCV

Pillow (PIL)

Visualisation

Matplotlib

Seaborn

Interface Web

Flask 

Outils

Google Colab (GPU)

Git & GitHub

Jupyter Notebook

📥 Installation
1️⃣ Cloner le projet
git clone https://github.com/Nour-el-houda34/Diabetic-Retinopathy.git
cd Diabetic-Retinopathy

2️⃣ Installer les dépendances
pip install -r requirements.txt


ou :

pip install tensorflow opencv-python pillow numpy pandas matplotlib seaborn scikit-learn flask django djangorestframework

3️⃣ Télécharger le dataset

Télécharger depuis Kaggle :

data/DiabeticBahia/

🔬 Pipeline de Prétraitement

L’image passe par :

Redimensionnement → 224×224

Normalisation → /255

Filtrage (optional)

Amélioration du contraste

Suppression du bruit

Recadrage circulaire du fond d’œil

🧠 Modèle de Deep Learning
🏛️ Architecture utilisée :

Base : ResNet50 (ou EfficientNet selon la version)

Pré-entraînement : ImageNet

Fine-tuning sur les 5 classes

Ajout de couches :

GlobalAveragePooling2D

Dense (256 neurons)

Dropout(0.3)

Dense(5, softmax)

⚙️ Compilateur :
optimizer = Adam(lr=0.0001)
loss = "sparse_categorical_crossentropy"
metrics = ["accuracy"]

🔁 Callbacks :

EarlyStopping

ReduceLROnPlateau

ModelCheckpoint

🏋️ Entraînement
python scripts/train_model.py


Ce script :

charge le dataset

applique les augmentations (rotation, zoom, flip, shift…)

entraîne le modèle

enregistre model.h5

🧪 Évaluation
python scripts/evaluate_model.py


Indicateurs utilisés :

Accuracy

Precision

Recall

F1-score

AUC

Matrice de confusion

📊 Résultats obtenus

Mesure	Valeur
Accuracy (validation)	89%
AUC	95.9%
F1-Score Healthy	94.3%
Loss	Stable, pas d’overfitting

🌐 Interface Web (Flask)
Fonctionnalités :

✔ Upload d’image
✔ Prédiction en temps réel
✔ Sauvegarde dans history.txt
✔ Affichage de l’image + classe
✔ API /predict (si version Django REST Framework)

Démo d'utilisation :

Ouvrir l'interface

Sélectionner une image

Cliquer sur Analyser

Le système affiche :

Classe prédite : Moderate DR (2)
Confiance : 91.4%

🚀 Lancer l’interface Web

Avec Flask :
python app.py


🧩 Améliorations Futures

Passage à EfficientNet B4 ou Swin Transformer

Déploiement sur Docker

API REST complète avec authentification

Interface React / Vue.js

Base de données pour historique réel

Rapport PDF automatique après analyse

Interprétation Grad-CAM (expliquer où le modèle regarde)

👥 Auteurs

BEN CHEIKHE Chaimae – Développement , Interface Graphique , Intégration

HAMIDI Nour El Houda – Deep Learning, Prétraitement, Data Exploration

TAIMOURIA El Bahia – Dataset, Entraînement, Gestion GitHub

📚 Projet « Deep Learning — ISI M2 — 2025–2026 »

Projet académique visant à appliquer les concepts de :

Vision par ordinateur

Deep Learning

Prétraitement d’images

Modèles CNN avancés

Déploiement d’un modèle IA