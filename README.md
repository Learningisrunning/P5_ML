# API de Prédiction de Tags pour Stack Overflow

Cette API utilise des modèles d'apprentissage automatique pour prédire des tags pertinents en fonction du contenu des questions postées sur Stack Overflow.

## Fonctionnalités

- **Prédiction des tags** : Fournissez une question en entrée, et obtenez des tags prédits.
- **REST API** : Interface simple pour interagir avec le modèle via des requêtes HTTP.

## Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/votre-utilisateur/stack-overflow-tag-predictor.git
   cd stack-overflow-tag-predictor

2. installer les dépendances : 
    pip install -r requirements.txt
    python -m nltk.downloader stopwords

3. lancer l'API
    uvicorn app.main:app --reload

4. Exemple : 
    Exemple de requête CURL : 

            curl -X POST "http://127.0.0.1:8000/predict" \
            -H "Content-Type: application/json" \
            -d '{
                "title": "How to implement a binary search in Python?",
                "body": "I need help with creating an efficient binary search function in Python."
            }'

    Exemple de réponse : 

            {
            "tags": ["python", "binary-search", "algorithm"]
            }
