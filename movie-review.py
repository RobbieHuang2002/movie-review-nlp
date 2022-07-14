import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
#load in the dataset of movie reviews
df = pd.read_csv("../Textfiles/moviereviews.tsv", sep='\t')

#check to see if there are any missing values in the review 
df.isnull().sum()


blanks = []
for i, lb, rv in df.itertuples():
    if rv.isspace():
        blanks.append(i)


X = df['review']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())]) 
text_clf.fit(X_train, y_train)
predictions = text_clf.predict(X_test)

print(accuracy_score(y_test,predictions))

