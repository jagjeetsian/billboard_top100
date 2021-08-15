# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 13:12:05 2021

@author: jagjeet
"""

from flask import Flask,request,render_template,redirect,url_for
import joblib
import spotipy
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials
client_credentials = SpotifyClientCredentials(client_id = '70ad902a9a224465a6e52af333c31cea', client_secret = 'fd060f8fb5d842c0a0a24f43c478ef93')
spotify = spotipy.Spotify(client_credentials_manager = client_credentials)



app=Flask(__name__)
model=joblib.load('pipeline.pkl')

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/proceed',methods=['POST'])
def proceed():
    artist=request.form['Artist']
    results = spotify.search(q='artist:' + artist , type='artist')
    items = results['artists']['items']
    if len(items) > 0:
        artist = items[0]
        d=items[0]['uri']
        results = spotify.artist_top_tracks(d)
    title_list=[]
    cover_list=[]
    features_list=[]
    for track in results['tracks'][:10]:
        title_list.append(track['name'])
        cover_list.append(track['album']['images'][0]['url'])
        features_list.append(spotify.audio_features(tracks=[track['id']])[0])
    joblib.dump([title_list,cover_list,features_list,artist],'details.pkl')
        
    return render_template('page2.html',output1=title_list,output2=cover_list)
@app.route('/check_again')
def check_again():
    return redirect(url_for("home"))
dict_genre={'alternative': 0, 'classical': 1, 'country': 2, 'edm': 3, 'jazz': 4, 'metal': 5, 'pop': 6, 'rap': 7, 'reggae': 8, 'rock': 9}
@app.route('/predict',methods=['POST'])
def predict():
    

    track=[int(i) for i in request.form.values()][0]
    
    details=joblib.load('details.pkl')
    title_list=details[0]
    cover_list=details[1]
    features_list=details[2]
    artist=details[3]
    features=features_list[track]
    danceability=features['danceability']
    energy=features['energy']
    loudness=features['loudness']
    speechiness=features['speechiness']
    acousticness=features['acousticness']
    instrumentalness=features['instrumentalness']
    mode=features['mode']
    liveness=features['liveness']
    valence=features['valence']
    tempo=features['tempo']
    common_genre=['alternative','classical','country', 'edm', 'jazz', 'metal', 'pop', 'rap', 'reggae', 'rock']    
    t=' '.join(artist['genres'])
    t=t.split()
    try:
        for i in t:
            if i in common_genre:
                genre=dict_genre[i]
                break
        
        
        model_features=[danceability,energy,loudness,speechiness,acousticness,mode,instrumentalness,liveness,valence,tempo,genre]

        final_features=([model_features])
        prediction=model.predict_proba(final_features)
        output=f"Song has {np.round(prediction[0][1],3)*100}% chanes of hitting the billboard top 100"
        #if prediction[0]==0:
        #    output='Song will not be in Billboard Top100'
        #else:
        #    output='Song might be in Billboard Top100'
    except:
        output="no genre found"
    return render_template('after.html', prediction_text=output)    

if __name__=='__main__':
    app.run(debug=True)