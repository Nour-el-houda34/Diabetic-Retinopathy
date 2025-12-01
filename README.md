#  Classification de la Rétinopathie Diabétique (Deep Learning + Interface Web)

**Projet** — Deep Learning · Flask · TensorFlow · ISI M2 (2025–2026)

---

##  Description

Système complet d’aide au diagnostic capable de classifier automatiquement des images de fond d’œil en **5 classes** de sévérité de la rétinopathie diabétique. Le projet inclut : prétraitement d’images, entraînement d’un modèle CNN (fine-tuning), évaluation et déploiement via une interface web Flask.

---

##  Objectif

Classer une image de fond d’œil en l’une des classes suivantes :

| Label | Classe                           |
| ----: | :------------------------------- |
|     0 | Healthy (Sain)                   |
|     1 | Mild DR (Léger)                  |
|     2 | Moderate DR (Modéré)             |
|     3 | Severe DR (Sévère)               |
|     4 | Proliferative DR (Proliférative) |

Le modèle renvoie la classe prédite et un score de confiance.

---

## Technologies

**Backend & Deep Learning**: Python 3.8+, TensorFlow/Keras, scikit-learn, pandas, numpy

**Traitement d’images**: OpenCV, Pillow

**Visualisation**: matplotlib, seaborn

**Interface Web**: Flask 

**Outils**: Google Colab (GPU), Git/GitHub, Jupyter Notebook

---

## Fonctionnalités principales

* Upload d’image via l’interface web
* Historique des analyses (`history.txt`)
* Endpoints API REST 
* Visualisation de la prédiction (affichage image + label)

---

##  Installation rapide

```bash
# Cloner le projet
git clone https://github.com/Nour-el-houda34/Diabetic-Retinopathy.git
cd Diabetic-Retinopathy

# Installer les dépendances (préférer un virtualenv)
pip install -r requirements.txt
```

Alternative (exemples de paquets si pas de requirements):

```bash
pip install tensorflow opencv-python pillow numpy pandas matplotlib seaborn scikit-learn flask
```

---

##  Structure du projet (suggestion)

```
Diabetic-Retinopathy/
├── README.md
├── requirements.txt
├── app.py                  # Flask app
├── scripts/
│   
│   ├── train_model.py
│   └── evaluate_model.py
├── notebooks/              # notebooks d'expérimentation
├── models/
│   └── best_model_diabetic.keras
├── static/
└── templates/
    └── index.html
```

---

##  Prétraitement (pipeline)

1. **Redimensionnement** → `224×224`
2. **Normalisation** → `/255`
3. **Amélioration** → equalize, CLAHE (optionnel)
4. **Filtrage** → suppression du bruit (GaussianBlur si besoin)
5. **Recadrage circulaire** → focaliser la région retina
6. **Augmentations** pour l'entraînement : rotation, flip, zoom, shift, brightness


---

##  Modèle

**Architecture** : ResNet50 (base) 
**Top layers** :

* GlobalAveragePooling2D
* Dense(256)
* Dropout(0.3)
* Dense(5, activation='softmax')

**Compilation** :

```py
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss = 'sparse_categorical_crossentropy'
metrics = ['accuracy']
```

**Callbacks** : `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`

---

##  Entraînement

```bash
python scripts/train_model.py --data_dir data/DiabeticBahia --epochs 60 --batch_size 32
```

Le script doit :

* charger et splitter le dataset (train/val/test)
* appliquer augmentations
* entraîner et sauvegarder `models/model.keras`

---

## Évaluation

```bash
python scripts/evaluate_model.py --model models/model.keras --data_dir data/DiabeticBahia
```

Indicateurs calculés : Accuracy, Precision, Recall, F1-score, AUC, matrice de confusion.

**Exemple de resultat résumé** :

| Mesure             | Valeur |
| ------------------ | -----: |
| Accuracy (val)     |    89% |
| AUC                |  95.9% |
| F1-score (Healthy) |  94.3% |


---

##  Interface Web (Flask)

**Lancer l’app localement** :

```bash
python app.py
# puis ouvrir http://127.0.0.1:5000
```
**Sauvegarde de l’historique** : `history.txt` (format CSV/JSON) 

---

## Exemple de sortie (UI)

* Classe prédite : **Moderate DR (2)**
* Confiance : **91.4%**
* Image affichée + Mode sombre (Selon  le choix des Utilisateurs )





---

## Auteurs

* **BEN CHEIKHE Chaimae** — Développement, Interface Graphique, Intégration
* **HAMIDI Nour El Houda** —Data Exploration, Prétraitement, Gestion Github
* **TAIMOURIA El Bahia** — Dataset, Entraînement, Phase de Test et Evaluation 

---

##  Licence

Copyright © 2025 _ Master Isi M2 _ 2025/2026
---




