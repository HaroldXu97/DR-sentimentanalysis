from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
from vectorizer import vect  # vectorizer

app = Flask(__name__)

# Classifier
cur_dir = os.path.dirname(__file__)
clf = pickle.load(
    open(os.path.join(cur_dir, 'pkl_objects', 'classifier_hash.pkl'), 'rb'))
db = os.path.join(cur_dir, 'reviews.sqlite')


def classify(document):
    '''
    input: document
    output: 
        - y: value of star
        - proba: list
    '''
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba_raw = clf.predict_proba(X)[0]
    proba = [round(p*100, 2) for p in proba_raw]
    return y, proba


def train(document, y):
    '''
    input: document, y
    '''
    X = vect.transform([document])
    clf.partial_fit(X, [y])


def sqlite_entry(path, document, y):
    '''
    input: path, document, y
    '''
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review, star, date)"
              " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()

# Flask


class ReviewForm(Form):
    yelpreview = TextAreaField('',
                               [validators.DataRequired(),
                                validators.length(min=15)])

# about


@app.route('/')
@app.route('/about')
def index():
    return render_template('index.html')

# projects


@app.route('/projects')
def projects():
    return render_template('projects.html')

# contact


@app.route('/contact')
def contact():
    return render_template('contact.html')

# PUBG
# report


@app.route('/projects/pubgreport')
def pubgreport():
    return render_template('pubgreport.html')

# SA
# report


@app.route('/projects/sareport')
def sareport():
    return render_template('sareport.html')

# form


@app.route('/projects/saform')
def saform():
    form = ReviewForm(request.form)
    return render_template('saform.html', form=form)

# result


@app.route('/projects/saresult', methods=['POST'])
def saresults():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['yelpreview']
        y, proba = classify(review)
        return render_template('saresult.html',
                               content=review,
                               prediction=y,
                               probability=proba)
    return render_template('saform.html', form=form)

# feedback


@app.route('/projects/safb', methods=['POST'])
def safb():
    feedback = request.form['feedback_button']
    review = request.form['review']
    y = int(feedback)
    train(review, y)
    # sqlite_entry(db, review, y)
    return render_template('safb.html')


if __name__ == '__main__':
    app.run(debug=True)
