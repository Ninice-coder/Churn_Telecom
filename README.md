# Analyse de Churn Télécom

Ce projet est une analyse de churn (attrition) dans le secteur des télécommunications en utilisant plusieurs algorithmes de machine learning. Le fichier Jupyter Notebook inclus (`telecom.ipynb`) guide à travers toutes les étapes nécessaires, de la préparation des données à l'évaluation des modèles.

## Contenu du Notebook

1. **Bibliothèque** : Importation des bibliothèques nécessaires pour l'analyse.
    - joblib
    - matplotlib
    - numpy
    - pandas
    - seaborn
    - scikit-learn
2. **Visualisation** : Exploration et visualisation des données pour mieux comprendre les tendances et les distributions.
3. **Préprocessing** : Nettoyage et préparation des données pour les algorithmes de machine learning.
4. **Entraînement** :
   - **Random Forest** : Application et évaluation du modèle Random Forest.
   - **SVM** : Application et évaluation du modèle Support Vector Machine.
   - **Gradient Boosting** : Application et évaluation du modèle Gradient Boosting.  
5. **Optimisation des Hyperparamètres** :
   - **GridSearch** : GridSearchCV est utilisé pour effectuer une recherche exhaustive sur un espace de grille spécifié d’hyperparamètres. Cela permet de trouver la meilleure combinaison d'hyperparamètres pour un modèle donné, améliorant ainsi sa performance.
   - **K-Fold Cross-Validation** : Cette technique divise les données en K sous-ensembles. Le modèle est entraîné sur K-1 de ces sous-ensembles et testé sur le sous-ensemble restant. Ce processus est répété K fois, chaque sous-ensemble servant exactement une fois de données de test. Cela permet de mieux évaluer la performance du modèle en utilisant différentes partitions des données.

## Comment utiliser ce projet
Clonez le dépôt sur votre machine locale.
   ```bash
   git clone https://github.com/Ninice-coder/Churn_Telecom.git
  
