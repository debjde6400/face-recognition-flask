
# A very simple Flask Hello World app for you to get started with...

import json
from flask import Flask, request
from flask_cors import CORS, cross_origin
from helpers import detect_face, recognize_face

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})

@app.route('/')
def hello_world():
    return 'Hello from Flask!'

# @app.route('/detect-unauth-objects', methods=['POST'])
# @cross_origin(origin='http://127.0.0.1:5500/index.html', headers=['Content-Type', 'Authorization'])
# def detect_unauth_objects():
#     request_data = request.get_json()
#     image = request_data['image']
#     violations_response = detect_unauthorized_objects(image)
#     return json.dumps(violations_response)

@app.route('/detect-faces', methods=['POST'])
@cross_origin(origin='http://127.0.0.1:5500/index.html', headers=['Content-Type', 'Authorization'])
def detect_faces_camera():
    request_data = request.get_json()
    image = request_data['image']
    task = request_data['task']
    response = {}

    if task == "REF_IMAGE":
        response = detect_face(image)

    elif task == "EXAM_RECOGNIZE":
        face_embedding = request_data['face_embedding']
        response = recognize_face(image, face_embedding)

    print(response)

    return json.dumps({ "response": response })

# main driver function
if __name__ == '__main__':
    app.run()
