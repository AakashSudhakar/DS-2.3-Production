from flask_restplus import Api, Resource, fields
from flask import Flask, request, jsonify
import numpy as np
from werkzeug.datastructures import FileStorage
from keras.models import model_from_json
import tensorflow as tf
import pickle

application = app = Flask(__name__)
api = Api(app, version='1.0', title='Cancer Data Classification', description='Aakash Sudhakar')
ns = api.namespace('CancerDataClassification', description='DS 2.3')

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
            data = fr.read()
            data = [data.strip("\n")]
        print(data[:3])
        with graph.as_default():
            out = model.predict(data)
        print("Out: ", out[0])
        pred = str(out[0])
        return {"Cancer Severity Classification [1-9]: ": pred}

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8000, debug=True)