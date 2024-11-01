import pandas as pd
import nltk
import ssl
import spacy
import re
from collections import Counter

#Data Cleaning: 

#Data getting:
df = pd.read_csv("data_from_stackoverflow.csv", sep=",")



#Set up : 
nltk.data.path.append('/Users/tomdumerle/nltk_data')

# Ignorer les erreurs de certificat
ssl._create_default_https_context = ssl._create_unverified_context

# Télécharger les stopwords
nltk.download('stopwords')

nlp = spacy.load('en_core_web_sm')


# création de la liste de base des stop words 
liste_default_stopwords = nltk.corpus.stopwords.words("english")


# création de la fonction le prétaitement du text
def preprocess_text(text):
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

custom_stopwords = get_custom_stopwords(df, n=5)

def get_the_complit_stop_word_list(df, liste_of_custom_stopwords):
    liste_of_custom_stopwords = [words[0] for words in liste_of_custom_stopwords ]
    liste_default_stopwords = nltk.corpus.stopwords.words("english")
    liste_of_stop_words = liste_default_stopwords + liste_of_custom_stopwords

    def filter_the_stopword(df_data,liste_of_stop_word):
        return [word for word in df_data if word not in liste_of_stop_word]

    df["Title_tokenized"] = df["Title_tokenized"].apply(lambda x : filter_the_stopword(x, liste_of_stop_words))
    df["Body_tokenized"] = df["Body_tokenized"].apply(lambda x : filter_the_stopword(x, liste_of_stop_words))

    return df

df_without_SW = get_the_complit_stop_word_list(df,custom_stopwords)


#Fonction de réduction de dimension des tags

def tag_dimension_reduction(colonne_tags, freq_min): 
    all_tags = df[colonne_tags].sum()
    words_frequences = Counter(all_tags)
    liste_tags_to_keep = {mot for mot, freq in words_frequences.items() if freq >=freq_min}
    df[colonne_tags] = df[colonne_tags].apply(lambda liste : [mot for mot in liste if mot in liste_tags_to_keep])

    return df[colonne_tags]


