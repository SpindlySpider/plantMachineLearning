from flask import Flask , render_template
from flask_socketio import SocketIO
import sys

# UPLOAD_FOLDER = '/app/docker_data' #the upload folder for recived files
# ai_module_path = "/app/imageClassification" # modules for docker
ai_module_path = "imageClassification"
UPLOAD_FOLDER = 'test'
model_name = "test"
image_name = "flower.jpg"
width = 250
height = 250
image_filepath = f"{UPLOAD_FOLDER}/{image_name}"
app = Flask(__name__,static_folder="")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
socketio = SocketIO(app)
sys.path.append(ai_module_path)

from imageClassification.utlities import *

@app.route("/")
def index():
    print(image_filepath)
    # return render_template("index.html",flower_image=f"../{image_filepath}")
    return render_template("index.html")

@socketio.on('file_upload')
def handle_upload(message):
    try:
        with open(f"{image_filepath}", "wb") as f:
            f.write(message)
    except Exception as e:
        print(e)
        
@socketio.on('predict')
def handle_predict():  # Rename the function to handle_predict
    prediction=predict_image(UPLOAD_FOLDER, model_name, image_filepath, width, height)
    print(prediction)
    socketio.emit("prediction",{"data":f"{prediction}"})


socketio.run(app=app,host='0.0.0.0', port=5000, debug=True)