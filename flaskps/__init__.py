from os import path
from flaskps.db import get_db
from flask import Flask, render_template, g, redirect, url_for, session
from flask_restful import Api
from flask_cors import CORS

from flaskps.config import Config
from flaskps.models.configuracion import Configuracion
from flaskps.resources import markov_lstm_implementation

from flaskps.resources.weights import Hip_Hop_weights, Freestyle_weights

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)
api = Api(app)

app.add_url_rule('/run/<string:word>/<string:style>', 'run_markovLstm', markov_lstm_implementation.do_word)

api.add_resource(Hip_Hop_weights, '/weights/hip_hop')
api.add_resource(Freestyle_weights, '/weights/freestyle')

@app.route("/")
def hello():
    #Configuracion.db = get_db
    return 'Hello boi. ready to rapapap doggy iou?'