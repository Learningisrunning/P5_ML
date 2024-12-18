import pandas as pd
import nltk
import ssl
import spacy
import re
from collections import Counter
from kedro.pipeline import Pipeline, node
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import mlflow
import mlflow.sklearn
from mlflow.pyfunc import load_model
from sklearn.metrics import hamming_loss
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import MultiLabelBinarizer


#Data Spliting: 
def split_data(df):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    return df_train, df_test
#Data Cleaning: 

# Ignorer les erreurs de certificat
ssl._create_default_https_context = ssl._create_unverified_context


def setup_nltk():
    nltk.data.path.append('/Users/tomdumerle/nltk_data')
    nltk.download('stopwords')

def setup_spacy() -> spacy.language.Language:
    return spacy.load('en_core_web_sm')

# création de la liste de base des stop words 
liste_default_stopwords = nltk.corpus.stopwords.words("english")


# création de la fonction le prétaitement du text
def preprocess_text(text, nlp):
    #Suppression de la ponctuation et des caractères spéciaux: 
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convertir en minuscules
    text = text.lower()
    # Tokenisation et lemmatisation
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in liste_default_stopwords]
    return tokens


#split des tags en liste
def process_tags_dataframe(df):
    """Récupération de la colonne tags, séparation en liste et remise en place dans le df original"""
    def split_in_list_text(text):
        tags = text.replace("<", "").split('>')
        return list(filter(None, tags))

    # Appliquer la fonction de transformation sur la colonne 'Tags'
    df["tags_liste"] = df["Tags"].apply(split_in_list_text)

    # Sélectionner uniquement les colonnes souhaitées
    return df[["Title_tokenized", "Body_tokenized", "tags_liste"]]


def get_custom_stopwords(df, n=5):
    """
    Calcule la fréquence des mots dans les colonnes 'Title_tokenized' et 'Body_tokenized'
    du DataFrame et retourne une liste des n mots les plus fréquents.

    :param df: DataFrame contenant les colonnes 'Title_tokenized' et 'Body_tokenized'
    :param n: Nombre de mots les plus fréquents à retourner
    :return: Liste des n mots les plus fréquents
    """
    freq_total = nltk.Counter()

    # Compter les mots dans 'Title_tokenized'
    for line in df["Title_tokenized"]:
        for word in line:
            freq_total[word] += 1

    # Compter les mots dans 'Body_tokenized'
    for line in df["Body_tokenized"]:
        for word in line:
            freq_total[word] += 1

    # Retourner les n mots les plus fréquents
    return freq_total.most_common(n)
    

def get_the_complit_stop_word_list(df, liste_of_custom_stopwords):
    liste_of_custom_stopwords = [words[0] for words in liste_of_custom_stopwords ]
    liste_default_stopwords = nltk.corpus.stopwords.words("english")
    liste_of_stop_words = liste_default_stopwords + liste_of_custom_stopwords

    def filter_the_stopword(df_data,liste_of_stop_word):
        return [word for word in df_data if word not in liste_of_stop_word]

    df["Title_tokenized"] = df["Title_tokenized"].apply(lambda x : filter_the_stopword(x, liste_of_stop_words))
    df["Body_tokenized"] = df["Body_tokenized"].apply(lambda x : filter_the_stopword(x, liste_of_stop_words))

    return df


#Fonction de réduction de dimension des tags

def tag_dimension_reduction(df, colonne_tags, freq_min): 
    all_tags = df[colonne_tags].sum()
    words_frequences = Counter(all_tags)
    liste_tags_to_keep = {mot for mot, freq in words_frequences.items() if freq >=freq_min}
    df[colonne_tags] = df[colonne_tags].apply(lambda liste : [mot for mot in liste if mot in liste_tags_to_keep])

    return df[colonne_tags]

#Mise en place de la pipeline pour le pré traitement du text : 

def load_data(url) -> pd.DataFrame:
    return pd.read_csv(url, sep=",")  # Chemin en brut ici


def run_pipeline_data_cleaning(url) -> pd.DataFrame:

    df = load_data(url)
    setup_nltk()
    nlp = setup_spacy()
    default_stopwords = nltk.corpus.stopwords.words("english")

    df["Title_tokenized"] = df["Title"].apply(lambda x: preprocess_text(x, nlp, default_stopwords))
    df["Body_tokenized"] = df["Body"].apply(lambda x: preprocess_text(x, nlp, default_stopwords))

    df = process_tags_dataframe(df)

    custom_stopwords = get_custom_stopwords(df, n=5)

    df = get_the_complit_stop_word_list(df, custom_stopwords)

    df["tags_liste"] = tag_dimension_reduction(df, "tags_liste", freq_min=100)

    return df

# Mise en place du BoW 

class BoWTransformer:
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vectorizer = CountVectorizer(max_features=self.max_features)
        
    def fit(self, Title):
        """Ajuste le vectorizer sur les données d'entraînement."""
        self.vectorizer.fit(Title)
        return self

    def transform(self, Title_body):
        """Transforme les données en BoW."""
        return self.vectorizer.transform(Title_body)

    def to_dataframe(self, data_transform):
        """Ajuste et transforme les données d'entraînement."""
        return pd.DataFrame(data_transform.toarray(),columns=self.vectorizer.get_feature_names_out())

# Fonction de prétraitement pour créer des colonnes tokenisées
def preprocess_for_bow(df):
    df["Title_tokenized_for_BoW"] = df["Title_tokenized"].apply(lambda x: ' '.join(x))
    df["Title_and_body_tokenized_for_BoW"] = (
        df["Title_tokenized"].apply(lambda x: ' '.join(x)) + 
        df["Body_tokenized"].apply(lambda x: ' '.join(x))
    )
    return df


def pipeline_transformation_to_BoW(df, df_name):

    df_boW = preprocess_for_bow(df)

    BoW_traitement = BoWTransformer(max_features=1000)

    X_fit = BoW_traitement.fit(df_boW["Title_tokenized_for_BoW"])

    X_transform = BoW_traitement.transform(df_boW["Title_and_body_tokenized_for_BoW"])

    df_boW_vf = BoW_traitement.to_dataframe(X_transform)

    df["tags"] = df["tags_liste"]

    return df_boW_vf
    

#Mise en place du ML:

def MultinomialNB_BoW_train(df_train):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    mlflow.start_run()

    X_train = df_train.drop("tags", axis = 1)
    y_train = df_train["tags"]

    pipeline_multinomialNB = Pipeline([
        ('model', MultinomialNB())
        ])

    pipeline_multinomialNB.fit(X_train, y_train)


    mlflow.sklearn.log_model(pipeline_multinomialNB, "MultinomialNB_BoW")

    model_name = "MultinomialNB_BoW"
    run_id = mlflow.active_run().info.run_id  # Utiliser le run_id de l'expérience en cours

    # Ajout dans le Model Registry
    mlflow.register_model(f"runs:/{run_id}/MultinomialNB_BoW", model_name)

    return pipeline_multinomialNB



def MultinomialNB_BoW_predict(df_test):

    X_test = df_test.drop("tags", axis = 1)
    y_test = df_test["tags"]
    model_name = "MultinomialNB_BoW"
    model_uri = f"models:/{model_name}/latest"
    pipeline_multinomialNB = load_model(model_uri)
    # Prédiction
    y_pred = pipeline_multinomialNB.predict(X_test)

    return y_pred, y_test

def accuracy(y_pred, y_test):

    # Binarisation des données
    mlb = MultiLabelBinarizer()
    y_test_bin = mlb.fit_transform(y_test)
    y_pred_bin = mlb.transform(y_pred)

    hamming_score = hamming_loss(y_test_bin, y_pred_bin)
    f1_micro_score = f1_score(y_test_bin, y_pred_bin, average="micro")
    f1_macro_score = f1_score(y_test_bin, y_pred_bin, average="macro")
    # Calcul du Jaccard score avec average='samples'
    jaccard_Score = jaccard_score(y_test_bin, y_pred_bin, average='samples')


    #Méthode d'évaluation de couverture des tags:

    recall_macro_score = recall_score(y_test_bin, y_pred_bin, average='macro')


    data = {
        "Hamming Loss" : hamming_score,
        "f1 macro" : f1_macro_score,
        "f1 micro" : f1_micro_score,
        "jaccard" : jaccard_Score,
        "recall macro" : recall_macro_score
    }

    data_score = pd.DataFrame(data, index=[0])

    # Convertir le DataFrame en dictionnaire
    metrics_dict = data_score.to_dict(orient='records')[0]  # 'records' permet d'obtenir une liste de dictionnaires

    # Enregistrer toutes les métriques dans MLflow
    mlflow.log_metrics(metrics_dict)

    return data_score




def training_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=load_data,
                inputs="params:url",
                outputs="raw_data",
                name="load_data_node",
            ),
            node(
                func=split_data,
                inputs="raw_data",
                outputs=["df_train", "df_test"],
                name="split_data_node",
            ),
            node(
                func=run_pipeline_data_cleaning,
                inputs="df_train",
                outputs="processed_data_train",
                name="data_cleaning_train_node",
            ),
            node(
                func=pipeline_transformation_to_BoW,
                inputs="processed_data_train",
                outputs="processed_data_boW_train",
                name="BoW_transformation_train_node",
            ),
            node(
                func=MultinomialNB_BoW_train,
                inputs="processed_data_boW_train",
                outputs="trained_model",
                name="train_model_node",
            ),
        ]
    )


def prediction_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=run_pipeline_data_cleaning,
                inputs="df_test",
                outputs="processed_data_test",
                name="data_cleaning_test_node",
            ),
            node(
                func=pipeline_transformation_to_BoW,
                inputs="processed_data_test",
                outputs="processed_data_boW_test",
                name="BoW_transformation_test_node",
            ),
            node(
                func=MultinomialNB_BoW_predict,
                inputs=["trained_model", "processed_data_boW_test"],
                outputs=["prediction", "test_set"],
                name="predict_model_node",
            ),
            node(
                func=accuracy,
                inputs=["prediction", "test_set"],
                outputs="accuracy_score",
                name="accuracy_node",
            ),
        ]
    )