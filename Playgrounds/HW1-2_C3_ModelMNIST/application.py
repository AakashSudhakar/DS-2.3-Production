# main.py (FLASK APP FOR MNIST IMAGE RECOGNITION USING COLAB FILE)

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from flask_restplus import Api, Resource, fields
from flask import Flask, request, jsonify
import numpy as np
from werkzeug.datastructures import FileStorage
from PIL import Image
from keras.models import model_from_json
import tensorflow as tf

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# TODO: Link Lines 18 and 23 with DB Setup on Firebase! 
creds = credentials.Certificate("flask-app.json")
firebase_admin.initialize_app(creds)

db = firestore.client()
print("PRINTING USERS")
users = db.collection(u'ds23_flask')
docs = users.get()

for doc in docs:
    print(u'{} -> {}'.format(doc.id, doc.to_dict()))

application = app = Flask(__name__)
api = Api(app, version='1.0', title='MNIST Classification', description='CNN for Mnist')
ns = api.namespace('Make_School', description='Methods')

single_parser = api.parser()
single_parser.add_argument("file", location="files", type=FileStorage, required=True)

model = load_model("my_model.h5")
graph = tf.get_default_graph()

@application.route("/")
def hello():
    docs = db.collection(u'ds23_flask').document(u'last_name_accessed')
    docs.set({
        u'time': time.time(),
    })

@ns.route('/prediction')
class CNNPrediction(Resource):
    '''Uploads your data to the CNN'''
    @api.doc(parser=single_parser, description='Upload an mnist image')
    def post(self):
        args = single_parser.parse_args()
        image_file = args.file
        image_file.save('image.png')
        img = Image.open('image.png')
        image_red = img.resize((28, 28))
        image = img_to_array(image_red)
        print(image.shape)
        x = image.reshape(1, 28, 28, 1)
        x = x/255
        # This is not good, because this code implies that the model will be
        # loaded each and every time a new request comes in.
        # model = load_model('my_model.h5')
        with graph.as_default():
            out = model.predict(x)
        print(out[0])
        print(np.argmax(out[0]))
        r = np.argmax(out[0])

        return {'prediction': str(r)}
        # TODO: Write filename, prediction result, and time accessed to DB after Firebase Setup!

if __name__ == '__main__':
    print("\n\nNumber of DB accesses: {}\n".format(23))
    app.run(host='0.0.0.0', port=8000)