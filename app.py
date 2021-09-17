from flask import Flask, render_template, request
import numpy as np
import joblib
import nltk   
nltk.download('all')
import os

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import Word


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict/',methods=['GET','POST'])
def submit():
    #HTML TO PY
    if request.method=="POST":
        tweet_text=request.form.get('tweet_text')
    
        try:
            prediction = preprocessDataAndPredict(tweet_text)
            #pass prediction to template
            return render_template('predict.html',
                                    tweet = tweet_text,
                                    prediction = prediction)

        except ValueError:
            return "Please Enter valid values"

        pass
    pass


def preprocessDataAndPredict(tweet_text):

    #pre processing steps like lower case, stemming and lemmatization
    tweet_text = tweet_text.lower()
    stop = stopwords.words('english')

    tweet_text = " ".join(x for x in tweet_text.split() if x not in stop)
    st = PorterStemmer()

    tweet_text = " ".join ([st.stem(word) for word in tweet_text.split()])
    tweet_text = " ".join ([Word(word).lemmatize() for word in tweet_text.split()])
 #open file
    file_model = open('sentiment.pkl', "rb")
    file_tfidf_vect = open('tfidf.pkl', "rb")

    #load the trained model
    trained_model = joblib.load(file_model)
    tfidf_vect = joblib.load(file_tfidf_vect)

    new_tweet_tfidf =  tfidf_vect.transform([tweet_text])

    prediction = trained_model.predict(new_tweet_tfidf)

    return prediction



if __name__ == '__main__':
	app.run(debug=True)






