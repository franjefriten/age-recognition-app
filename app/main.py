import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms, models
from flask import Flask, Response, redirect, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from transformers import AutoModelForImageClassification, AutoTokenizer
import os
import pathlib

# model paths
local_path = "./models/vit-age-classifier"

# load age classifier model
model_name = "nateraw/vit-age-classifier"
age_model = AutoModelForImageClassification.from_pretrained(model_name, cache_dir=local_path)

# Load YOLO face detection model
yolo_model = YOLO("./models/yolov8n.pt")


# transform para preprocesar imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

app = Flask(__name__)

# Upload file
app.config['UPLOAD'] = "app/static/upload"

# Function to process video frame-by-frame
def process_video():
    cap = cv2.VideoCapture(0)  # Abrir webcam
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = yolo_model(frame)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Vértices de la caja
                face = frame[y1:y2, x1:x2]  # Obtener la cara
                
                # Convert to PIL image for age prediction
                face_pil = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_pil = transform(Image.fromarray(face_pil)).unsqueeze(0)

                # Predecir
                with torch.no_grad():
                    prediction = age_model(face_pil)
                    logits = prediction.logits  
                    age_category = torch.argmax(logits, dim=1).item()  
                    age_label = f"{age_category * 10}-{(age_category + 1) * 10}"

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, age_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Encode the frame and return it as a byte stream
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()


def run_yolo_and_predict_age(img):
    img = cv2.imread(filename=img)

    results = yolo_model(source=img)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = img[y1:y2, x1:x2]
            face_pil = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_pil = transform(Image.fromarray(face_pil)).unsqueeze(0)

            # Convert to PIL image for age prediction
            face_pil = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_pil = transform(Image.fromarray(face_pil)).unsqueeze(0)

            with torch.no_grad():
                prediction = age_model(face_pil)
                logits = prediction.logits
                age_category = torch.argmax(logits, dim=1).item()

    return f"{age_category * 10}-{(age_category + 1) * 10}"


################
## APP ROUTES ##
################


@app.route("/", methods=["GET"])
def root():
    return redirect(url_for("home"))

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/image-predict", methods=["POST", "GET"])
def image_predict():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        if pathlib.Path(filename).suffix in ['.jpg', '.png']:
            img = os.path.join(app.config['UPLOAD'], filename)
            img_url = f"upload/{filename}"
            file.save(img)
            prediction = run_yolo_and_predict_age(img=img)
            return render_template("image-predict.html", img=img_url, pred=prediction)
    return render_template("image-predict.html")

# Flask route to serve the video stream
@app.route("/video-feed")
def video_feed():
    return Response(process_video(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)