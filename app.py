# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 13:12:05 2021

@author: jagjeet
"""

from flask import Flask,request,render_template,redirect,url_for
import numpy as np
import joblib


sc=joblib.load('scaler.pkl')
app=Flask(__name__)
model=joblib.load('decision_tree_boosted.pkl')

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/check_again')
def check_again():
    return redirect(url_for("home"))
dict_genre={'alternative': 0, 'classical': 1, 'country': 2, 'edm': 3, 'jazz': 4, 'metal': 5, 'pop': 6, 'rap': 7, 'reggae': 8, 'rock': 9}
@app.route('/predict',methods=['POST'])
def predict():
    

    danceability=float(request.form['Danceability'])
    energy=float(request.form['Energy'])
    loudness=float(request.form['Loudness'])
    speechiness=float(request.form['Speechiness'])
    acousticness=float(request.form['Acousticness'])
    liveness=float(request.form['Liveness'])
    valence=float(request.form['Valence'])
    tempo=float(request.form['Tempo'])
    genre=int(dict_genre[request.form['Genre']])
    features=[danceability,energy,loudness,speechiness,acousticness,liveness,valence,tempo,genre]
    final_features=sc.transform([features])
    prediction=model.predict(final_features)
    if prediction[0]==0:
        output='Song will not be in Billboard Top100'
    else:
        output='Song might be in Billboard Top100'
    return render_template('after.html', prediction_text=output)    

if __name__=='__main__':
    app.run(debug=True)