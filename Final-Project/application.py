# from keras.preprocessing.image import img_to_array
# from keras.models import load_model
from flask_restplus import Api, Resource, fields
from flask import Flask, request, jsonify
import numpy as np
from werkzeug.datastructures import FileStorage
# from PIL import Image
from keras.models import model_from_json
import tensorflow as tf
import pickle
# # import firebase_admin
# from firebase_admin import credentials, firestore
# import datetime

# cred = credentials.Certificate("./ServiceAccountKey.json")
# firebase_app = firebase_admin.initialize_app(cred)
# db = firestore.client()

application = app = Flask(__name__)
api = Api(app, version='1.0', title='Cancer Data Classification', description='Aakash Sudhakar')
ns = api.namespace('Cancer Data Classification', description='DS 2.3')

single_parser = api.parser()
single_parser.add_argument('file', location='files',
  type=FileStorage, required=True)

# model = load_model('my_model.pkl')
model = pickle.load(open("my_model.pkl", "rb"))
graph = tf.get_default_graph()

@ns.route('/prediction')
class SVCPrediction(Resource):
    """ Uploads data to SVM. """
    @api.doc(parser=single_parser, description="Upload a clinical record.")
    def post(self):
        args = single_parser.parse_args()
        record = args.file
        record.save("last_record.txt")
        with open("last_record.txt") as fr:
            data = [fr.read().strip("\n")]
        print(data[:3])
        with graph.as_default():
            out = model.predict(data)
        print("Out: ", out[0])
        pred = str(out[0])
        return {"Cancer Severity Classification [1-9]: ": pred}
        
# class CNNPrediction(Resource):
#   """Uploads your data to the CNN"""
#   @api.doc(parser=single_parser, description='Upload an mnist image')
#   def post(self):
#     args = single_parser.parse_args()
#     image_file = args.file
#     image_file.save('last_image.png')
#     img = Image.open('last_image.png')
#     print("size: ", img.size)
#     image_red = img.resize((150, 150)).convert('RGB')
#     image = img_to_array(image_red)
#     print("1. Image shape:", image.shape)
#     x = image.reshape(1, 150, 150, 3)
#     x = x/255
#     with graph.as_default():
#       out = model.predict(x)
#     print("Out:", out[0][0])
#     pred = str(round(out[0][0] * 100))
#     savePredictionToDB(pred)
#     return {'Likelyhood of Pneumonia:': pred + " percent"}

# def savePredictionToDB(pred):
#   time = datetime.datetime.now()
#   title = "pred " + time.strftime("%I:%M%p on %B %d, %Y")
#   doc_ref = db.collection(u'userlogs').document(title)
#   doc_ref.set({
#     u'prediction': pred,
#     u'time': time,
#   })

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8000, debug=True)