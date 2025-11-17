from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def test_tfidf_lr_smoke():
    X = ["hello world", "goodbye world"]
    y = [0, 1]
    pipe = Pipeline([('tfidf', TfidfVectorizer()), ('lr', LogisticRegression())])
    pipe.fit(X, y)
    pred = pipe.predict(["hello"])
    assert pred[0] == 0