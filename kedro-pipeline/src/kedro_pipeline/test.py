import pytest
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB

from pipelines.predict_pipeline.nodes import (
    #setup_nltk,
    setup_spacy,
    preprocess_text,
    get_custom_stopwords,
    get_the_complit_stop_word_list,
    BoWTransformer,
    pipeline_transformation_to_BoW,
    MultinomialNB_BoW_predict

)

# Test pour le prétraitement du texte
def test_preprocess_text():
    #setup_nltk()
    nlp = setup_spacy()
    text = "Hello, this is a simple test for text preprocessing!"
    default_stopwords = ["this", "is", "a", "for"]

    tokens = preprocess_text(text, nlp, default_stopwords)

    assert isinstance(tokens, list)
    assert "hello" in tokens
    assert "test" in tokens
    assert "this" not in tokens

# Test pour get_custom_stopwords
def test_get_custom_stopwords():
    df = pd.DataFrame({
        "Title_tokenized": [["word1", "word2", "word3"], ["word1", "word4", "word5"]],
        "Body_tokenized": [["word3", "word4", "word6"], ["word1", "word7", "word8"]]
    })
    custom_stopwords = get_custom_stopwords(df, n=2)

    assert len(custom_stopwords) == 2
    assert custom_stopwords[0][0] == "word1"

# Test pour get_the_complit_stop_word_list
def test_get_the_complit_stop_word_list():
    df = pd.DataFrame({
        "Title_tokenized": [["word1", "word2", "word3"]],
        "Body_tokenized": [["word4", "word5", "word6"]]
    })
    custom_stopwords = [("word1", 3), ("word2", 2)]

    result_df = get_the_complit_stop_word_list(df, custom_stopwords)

    assert "word1" not in result_df["Title_tokenized"].iloc[0]

# Test pour BoWTransformer
def test_bow_transformer():
    texts = ["text one", "text two", "another text"]
    bow = BoWTransformer(max_features=5)
    bow.fit(texts)

    transformed = bow.transform(texts)
    df = bow.to_dataframe(transformed)

    assert df.shape[0] == len(texts)
    assert len(bow.vectorizer.get_feature_names_out()) <= 5

# Test pour pipeline_transformation_to_BoW
def test_pipeline_transformation_to_bow():
    df = pd.DataFrame({
        "Title_tokenized": [["word1", "word2"], ["word3", "word4"]],
        "Body_tokenized": [["word5", "word6"], ["word7", "word8"]]
    })
    bow = BoWTransformer(max_features=10)
    bow.fit(df["Title_tokenized"].apply(lambda x: " ".join(x)))

    transformed_df = pipeline_transformation_to_BoW(df, bow)

    assert not transformed_df.empty

# Test pour MultinomialNB_BoW_predict
def test_multinomialnb_bow_predict():
    X = [[1, 0, 1], [0, 1, 1]]
    y = [["tag1"], ["tag2"]]

    mlb = MultiLabelBinarizer()
    y_bin = mlb.fit_transform(y)

    model = MultinomialNB()
    model.fit(X, y_bin)

    y_pred = MultinomialNB_BoW_predict(model, [[1, 0, 1]], mlb)

    assert y_pred == ["tag1"]
