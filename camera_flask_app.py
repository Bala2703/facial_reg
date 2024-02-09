# import pathlib
# from flask import Flask, render_template, Response, request
# import cv2
# import os
# import numpy as np
# from threading import Event
# import uuid

# global grey, camera, dataset_name, stop_event
# grey = 0
# dataset_name = None  
# stop_event = Event() 
# camera = None 

# try:
#     directory = "dataset"
#     os.makedirs(directory)

# except OSError as error:
#     pass

# app = Flask(__name__, template_folder='./templates')

# def gen_frames():
#     global grey, camera, dataset_name, stop_event, dataset_uuid
#     count = 0

#     while not stop_event.is_set():
#         success, frame = camera.read()

#         if success and grey:  # Only process frames when grey is True
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             path = str(pathlib.Path(__file__).parent.resolve()) + '/haarcascade_frontalface_default.xml'
#             face_detector = cv2.CascadeClassifier(path)
#             faces = face_detector.detectMultiScale(gray, 1.3, 5)
#             for (x, y, w, h) in faces:
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#                 if dataset_name is not None:
#                     # Adjust the path to include the dataset name
#                     image_path = os.path.join("dataset", str(dataset_name), "User." + str(dataset_uuid) + "." + str(count)+".jpg")

#                     # Save the image
#                     success = cv2.imwrite(image_path, gray[y:y + h, x:x + w])
#                     if success:
#                         count += 1
#                         print("Image saved successfully.")
#                     else:
#                         print("Failed to save image.")
#                 else:
#                     print("Dataset name is None. Handle this case according to your application logic.")

#                 if count >= 50:
#                     stop_event.set()  # Stop the loop

#             try:
#                 ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
#                 frame = buffer.tobytes()
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#             except Exception as e:
#                 pass

#         else:
#             pass

#     if camera.isOpened():
#         camera.release()
#     cv2.destroyAllWindows()

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     global camera, grey
#     if camera is None or not camera.isOpened():
#         camera = cv2.VideoCapture(0)
#     grey = 1
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/requests', methods=['POST', 'GET'])
# def tasks():
#     global grey, dataset_name, stop_event, dataset_uuid
#     if request.method == 'POST':
#         if request.form.get('create') == 'Create_dataset':
#             dataset_name = request.form.get('dataset_name')

#             if dataset_name and dataset_name.strip():
#                 dataset_folder_path = os.path.join("dataset", dataset_name)
#                 try:
#                     os.makedirs(dataset_folder_path, exist_ok=True)
#                     dataset_uuid = uuid.uuid1()
#                     print(f"Dataset folder '{dataset_name}' created successfully.")
#                     grey = not grey
#                     stop_event.clear()  
#                 except OSError as e:
#                     print(f"Error creating dataset folder '{dataset_name}': {e}")
#             else:
#                 print("Dataset name is empty or contains only whitespace. Handle this case according to your application logic.")

#     elif request.method == 'GET':
#         return render_template('index.html')

#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run()

import pathlib
from flask import Flask, render_template, Response, request, redirect, url_for, flash
import cv2
import os
import numpy as np
import uuid
from threading import Event
import firebase_admin
from firebase_admin import credentials, db
from PIL import Image
import pyrebase
from requests.exceptions import HTTPError

config = {
  "apiKey": "AIzaSyAhnt3bVjfG7W_e8tmR7v2GcFLRTVBtDSE",
  "authDomain": "attendance-71cd0.firebaseapp.com",
  "databaseURL": "https://attendance-71cd0-default-rtdb.firebaseio.com/",
  "storageBucket":  "attendance-71cd0.appspot.com",
  "serviceAccount": "attendance-71cd0-firebase-adminsdk-s5fvg-679f8dc269.json"
}
cred = credentials.Certificate("attendance-71cd0-firebase-adminsdk-s5fvg-679f8dc269.json")
firebase_admin.initialize_app(cred, {'databaseURL': 'https://attendance-71cd0-default-rtdb.firebaseio.com/'})
firebase = pyrebase.initialize_app(config)
global grey, camera, dataset_name, stop_event, dataset_uuid, user, auth_done, dataset_done
grey = 0
dataset_name = None
stop_event = Event()
dataset_uuid = None
camera = None
auth_done = False
dataset_done = False
try:
    directory = "dataset"
    os.makedirs(directory)

except OSError as error:
    pass

app = Flask(__name__, template_folder='./templates')

path = 'dataset'
trainer_directory = 'trainer'
if not os.path.exists(trainer_directory):
    os.makedirs(trainer_directory)

recognizer = cv2.face.LBPHFaceRecognizer_create()
cascadePath = str(pathlib.Path(__file__).parent.resolve()) + '/haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(cascadePath)
 
def gen_frames():
    global grey, camera, dataset_name, stop_event, dataset_uuid
    count = 0

    while not stop_event.is_set():
        success, frame = camera.read()

        if success and grey:  
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            path = str(pathlib.Path(__file__).parent.resolve()) + '/haarcascade_frontalface_default.xml'
            face_detector = cv2.CascadeClassifier(path)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                if dataset_name is not None:
                    image_path = os.path.join("dataset", str(dataset_name), "User." + str(dataset_uuid) + "." + str(count) + ".jpg")

                    success = cv2.imwrite(image_path, gray[y:y + h, x:x + w])
                    if success:
                        count += 1
                        print("Image saved successfully.")
                        store_mapping_in_firebase(str(dataset_name), str(dataset_uuid))
                    else:
                        print("Failed to save image.")
                else:
                    print("Dataset name is None. Handle this case according to your application logic.")

                if count >= 30:
                    stop_event.set()
                    train_face_recognizer()

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            except Exception as e:
                pass

        else:
            pass
    
    if camera.isOpened():
        camera.release()

def gen_recon_frames():
    global camera
    db = firebase.database()
    db.child("Students").child("Bala")
    recognizer.read('trainer/trainer.yml')

    cascadePath = str(pathlib.Path(__file__).parent.resolve()) + '/haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX

    minW = 0.1 * camera.get(3)
    minH = 0.1 * camera.get(4)

    dataset_path = 'dataset'
    names = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
    names.insert(0, 'None')
    while True:
        ret, img = camera.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if confidence < 100:
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
        try:
            ret, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()
            db.child("Students").child("Bala").update({"name": "Bala Kumar", "Status": "Present"})
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

        except Exception as e:
            pass

def store_mapping_in_firebase(folder_name, uuid):
    ref = db.reference('/folder_name_uuid_mapping')
    ref.update({folder_name: uuid})

def train_face_recognizer():
    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = get_images_and_labels(path)
    id_mapping = {uuid: i for i, uuid in enumerate(np.unique(ids))}
    int_ids = [id_mapping[uuid] for uuid in ids]
    int_ids = np.array(int_ids, dtype=np.int32)

    recognizer.train(faces, int_ids)
    recognizer.write(os.path.join(trainer_directory, 'trainer.yml'))
    save_uuid_mapping_to_realtime_db(id_mapping)
    dataset_done = True
    print("\n [INFO] Face recognition model trained successfully.")

def save_uuid_mapping_to_realtime_db(uuid_mapping):
    string_uuid_mapping = {str(uuid): value for uuid, value in uuid_mapping.items()}
    ref = db.reference('/uuid_folder_mapping')
    ref.set(string_uuid_mapping)

def get_images_and_labels(path):
    image_paths = []

    for root, dirs, files in os.walk(path):
        for directory in dirs:
            dir_path = os.path.join(root, directory)
            image_paths.extend([os.path.join(dir_path, f) for f in os.listdir(dir_path)])

    face_samples = []
    ids = []

    for image_path in image_paths:
        PIL_img = Image.open(image_path).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')

        id_str = os.path.split(image_path)[-1].split(".")[1]
        id = uuid.UUID(id_str)
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return face_samples, ids

@app.route('/', methods=['POST', 'GET'])
def login():
    global user, auth_done
    if request.method == 'POST':
        if request.form.get('login_btn') == 'Login':
            auth = firebase.auth()
            try:
                email = request.form.get('email')
                password = request.form.get('password')
                user = auth.sign_in_with_email_and_password(email, password)
                auth_done = True
                return redirect(url_for('tasks'))
            except HTTPError as e:
                flash("Check your email and password.")
                auth_done = False
                return render_template('login.html')
    return render_template('login.html')
    	 

@app.route('/video_feed')
def video_feed():
    global camera, grey
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(3, 640)
        camera.set(4, 480)
    grey = 1
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recon_feed')
def recon_feed():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(3, 640)
        camera.set(4, 480)
    print("recon")
    return Response(gen_recon_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recon_request')
def recon_request():
    return render_template('recon.html')

@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global grey, dataset_name, stop_event, dataset_uuid, camera
    if auth_done == True:
        if request.method == 'POST':
            if request.form.get('create') == 'Create_dataset':
                dataset_name = request.form.get('dataset_name')

                if dataset_name and dataset_name.strip():
                    dataset_uuid = uuid.uuid1()
                    dataset_folder_path = os.path.join("dataset", dataset_name)
                    try:
                        os.makedirs(dataset_folder_path, exist_ok=True)
                        print(f"Dataset folder '{dataset_name}' created successfully.")
                        grey = not grey
                        stop_event.clear() 
                    except OSError as e:
                        print(f"Error creating dataset folder '{dataset_name}': {e}")
                else:
                    print("Dataset name is empty or contains only whitespace. Handle this case according to your application logic.")
            elif request.form.get('start_btn'):
                return redirect(url_for('recon_request'))

        elif request.method == 'GET':
            return render_template('index.html')

        return render_template('index.html')
    

if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host="127.0.0.1",port=5000)
