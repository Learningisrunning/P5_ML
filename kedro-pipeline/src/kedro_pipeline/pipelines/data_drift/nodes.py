from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DataDriftTable
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
import nltk
import ssl
import spacy
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline

def split_the_data(df):

        # Convertir la colonne 'CreationDate' en datetime
    df['CreationDate'] = pd.to_datetime(df['CreationDate'])

    # Extraire uniquement l'année et le mois sous forme de Period (année-mois)
    df['CreationDate'] = df['CreationDate'].dt.to_period('M')

    # Convertir la colonne 'CreationDate' en datetime (timestamp) avant de faire des calculs
    df['CreationDate'] = df['CreationDate'].dt.to_timestamp()

    # Trouver la date la plus récente
    date_max = df['CreationDate'].max()

    # Calculer la date limite pour les 12 derniers mois
    date_limite = date_max - pd.DateOffset(months=12)

    # Filtrer les 12 derniers mois
    df_recents = df[df['CreationDate'] > date_limite]

    # Filtrer tout le reste
    df_reste = df[df['CreationDate'] <= date_limite]

    # Créer une colonne 'mois' qui contient le mois (de 1 à 12)
    df_for_the_months = df_recents
    df_for_the_months['mois'] = df_for_the_months['CreationDate'].dt.month
   
    return df_reste, df_recents, df_for_the_months


#Split the months:

def split_the_month(df_recents):

    list_of_datasets = []

    # Découper le dataframe en 12 datasets (un pour chaque mois)
    for mois in range(1, 13):
        # Filtrer le DataFrame pour chaque mois et ajouter à la liste
       
        df_recents_month = df_recents[df_recents['mois'] == mois]
        if "mois" in df_recents_month.columns:
            df_recents_month = df_recents_month.drop("mois", axis=1)
        else:
            print("La colonne 'mois' n'existe pas dans df_recents_month.")

        list_of_datasets.append(df_recents_month)
    
    return list_of_datasets


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
def preprocess_text(text, nlp, liste_default_SW):
    #Suppression de la ponctuation et des caractères spéciaux: 
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convertir en minuscules
    text = text.lower()
    # Tokenisation et lemmatisation
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in liste_default_SW]
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

def load_data(data: pd.DataFrame) -> pd.DataFrame:
    return data

def run_pipeline_data_cleaning(data) -> pd.DataFrame:

    df = data
    setup_nltk()
    nlp = setup_spacy()
    default_stopwords = nltk.corpus.stopwords.words("english")

    df["Title_tokenized"] = df["Title"].apply(lambda x: preprocess_text(x, nlp, default_stopwords))
    df["Body_tokenized"] = df["Body"].apply(lambda x: preprocess_text(x, nlp, default_stopwords))

    df = process_tags_dataframe(df)

    custom_stopwords = get_custom_stopwords(df, n=5)

    df = get_the_complit_stop_word_list(df, custom_stopwords)

    df["tags_liste"] = tag_dimension_reduction(df, "tags_liste", freq_min=100)

    df.drop("tags_liste", axis=1, inplace=True)
    df['Title_tokenized'] = df['Title_tokenized'].apply(lambda x: ' '.join(x))
    df['Body_tokenized'] = df['Body_tokenized'].apply(lambda x: ' '.join(x))

    df.dropna(inplace=True)

    return df

def add_the_month(df,df_for_month):

    df["mois"] = df_for_month["mois"]
    df.dropna(inplace=True)

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


def pipeline_transformation_to_BoW(df_rest, df_recents, df_for_the_month):

    df_boW_rest = preprocess_for_bow(df_rest)
    df_boW_recents = preprocess_for_bow(df_recents)

    BoW_traitement = BoWTransformer(max_features=1000)

    X_fit = BoW_traitement.fit(df_boW_rest["Title_tokenized_for_BoW"])

    X_transform_rest = BoW_traitement.transform(df_boW_rest["Title_and_body_tokenized_for_BoW"])
    X_transform_recents = BoW_traitement.transform(df_boW_recents["Title_and_body_tokenized_for_BoW"])

    df_boW_vf_rest = BoW_traitement.to_dataframe(X_transform_rest)
    df_boW_vf_recents = BoW_traitement.to_dataframe(X_transform_recents)

    df_boW_vf_rest.dropna(inplace=True)

    df_boW_vf_recents["mois"] = df_for_the_month["mois"]
    df_boW_vf_recents.dropna(inplace=True)


    return df_boW_vf_rest, df_boW_vf_recents



def monitor_data_drift(reference_data, current_data_list):
    """
    Monitors data drift between reference data and multiple current datasets (e.g., monthly datasets).
    
    Args:
        reference_data (pd.DataFrame): The dataset used for training.
        current_data_list (list of pd.DataFrame): List of datasets used for testing or in production (e.g., monthly data).
        output_path (str): Path to save the generated report.
    """
    
    output_path = "data/P5"
    # Vérifier si le répertoire existe, sinon le créer
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Répertoire créé: {output_path}")
        
    # Loop through each dataset in current_data_list
    for i, current_data in enumerate(current_data_list):
        # Create a report with Evidently
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)
    
        # Generate the output file name for each report (e.g., month_1.html, month_2.html, ...)
        output_file = os.path.join(output_path, f"data_drift_month_{i+1}.html")
        
        # Save the report as HTML
        report.save_html(output_file)
        print(f"Data drift report for month {i+1} saved at {output_file}")
