from flask import Flask,render_template,request
import google.generativeai as genai
import random
import os
from dotenv import load_dotenv
import joblib
import sklearn
from textblob import TextBlob
from transformers import pipeline
#import torch
#from diffusers import StableDiffusionPipeline, DiffusionPipeline,DPMSolverMultistepScheduler
#from web3 import Web3 # Too large for Render.com, using JS Web3 instead

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)
app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def index():
    return(render_template("index.html"))

@app.route("/financial_FAQ",methods=["GET","POST"])
def financial_FAQ():
    return(render_template("financial_FAQ.html"))

@app.route("/makersuite",methods=["GET","POST"])
def makersuite():
    model = genai.GenerativeModel('gemini-1.5-flash')
    q = request.form.get("q")
    r = model.generate_content(q)
    return(render_template("makersuite.html", r=r.text))

jokes = [
    "Why couldn't encik order McSpicy upsize? Because he's a regular.",
    "Why trees unlucky? Cos they sway.",
    "Which noodle is the heaviest? Wanton (one-tonne) noodle.!",
    "How you come here? I take bus 11 (2 legs)",
    "BreadTalk and Kopitiam, who is more talkative? BreadTalk, because Bread talk, Kopi tiam.",
    "What did the Singapore Airlines air stewardess say when a passenger blocked her way? SQ me!",
    "Which is the tallest building in Singapore? – The National Library, because it has many stories."
]

@app.route("/joke",methods=["GET","POST"])
def joke():
    joke = random.choice(jokes)
    return render_template('joke.html', joke=joke)

@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    predictionmodel = joblib.load('model/dbs_model.jl')
    predicted_price = None
    if request.method == 'POST':
        try:
            sgd_value = float(request.form['sgd_value'])
            predicted_price = predictionmodel.predict([[sgd_value]])
            return render_template('prediction.html', predicted_price=predicted_price[0][0])

        except Exception as e:
            print("Error during prediction:", e)
            return "Bad Request: " + str(e), 400

    return render_template('prediction.html', predicted_price=predicted_price)

@app.route('/sentiment', methods=["GET", "POST"])
def sentiment():
    sentiment_textblob = None
    sentiment_transformers = None
    if request.method == 'POST':
        try:
            text = str(request.form['text'])
            print(text)
            sentiment_textblob = TextBlob(text).sentiment
            #sentiment_transformers = pipeline("sentiment-analysis", device="mps", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")(text)
            sentiment_transformers = pipeline("sentiment-analysis", device="cpu", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")(text)
            sentiment_transformers = f"label: {sentiment_transformers[0]['label']}, score: {sentiment_transformers[0]['score']}"

        except Exception as e:
            print("Error during prediction:", e)
            return "Bad Request: " + str(e), 400
    
    return render_template('sentiment.html', sentiment_textblob=sentiment_textblob, sentiment_transformers=sentiment_transformers)

@app.route('/transfer', methods=["GET", "POST"])
def transfer():
    return render_template('transfer.html')

if __name__ == "__main__":
    app.run()