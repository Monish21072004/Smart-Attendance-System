import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash, send_from_directory, \
    Response
from PIL import Image
from werkzeug.utils import secure_filename
from torchvision.models import ResNet50_Weights
from flask_sqlalchemy import SQLAlchemy
import cv2  # For live camera and face detection

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure secret key

# Set the uploads folder to your absolute path
app.config['UPLOAD_FOLDER'] = r'C:\Users\Monish V\OneDrive\Documents\RANDOM_PROJECTS\Face_detction\uploads'

# Configure MySQL connection using your credentials
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Root%40123@localhost:3306/smart_attendance'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


# -------------------------
# Database Models
# -------------------------
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)  # Plain text for demo purposes
    role = db.Column(db.Enum('teacher', 'student'), nullable=False)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, server_default=db.func.now())


class Attendance(db.Model):
    __tablename__ = 'attendance'
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, nullable=False)
    attendance_timestamp = db.Column(db.DateTime, server_default=db.func.now())
    image_path = db.Column(db.String(255))  # This stores only the unique filename


# Uncomment the next line on first run to create tables, then comment it out.
# db.create_all()

# -------------------------
# Ensure the uploads folder exists
# -------------------------
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


# -------------------------
# Route to serve uploaded files
# -------------------------
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# -------------------------
# Dedicated Image View Routes
# -------------------------
@app.route('/teacher/view_image/<path:filename>')
def teacher_view_image(filename):
    filename = os.path.basename(filename)
    return render_template("view_image_teacher.html", filename=filename)


@app.route('/student/view_image/<path:filename>')
def student_view_image(filename):
    filename = os.path.basename(filename)
    return render_template("view_image_student.html", filename=filename)


# -------------------------
# Face Recognition Setup
# -------------------------
# For this app, we use all detected classes (from ImageFolder)
DATASET_DIR = r"C:\Users\Monish V\OneDrive\Documents\RANDOM_PROJECTS\Face_detction\Dataset\Original Images\Original Images"
dataset = ImageFolder(root=DATASET_DIR)
class_names = dataset.classes
num_classes = len(class_names)
print("Detected classes:", class_names)


class FaceClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FaceClassifier, self).__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        # Replace the final fully-connected layer with our own classifier
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


model = FaceClassifier(num_classes)

# Load state dict and remap keys to fix mismatches
state_dict = torch.load("face_classifier3.pth", map_location=torch.device('cpu'))
new_state_dict = {}
for key, value in state_dict.items():
    # Replace unexpected keys "backbone.fc.1." with expected "backbone.fc."
    new_key = key.replace("backbone.fc.1.", "backbone.fc.")
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# -------------------------
# Live Camera Face Detection with Multi-face Support
# -------------------------
def gen_frames():
    # Initialize the video capture object (0 for default camera)
    cap = cv2.VideoCapture(0)
    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert frame to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect multiple faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame in byte format for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


# New route to serve the live video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Route to render the live view page (create a corresponding template "live.html")
@app.route('/live')
def live():
    return render_template("live.html")


# -------------------------
# Routes
# -------------------------
@app.route('/')
def index():
    return render_template("login.html")


# -------------------------
# Teacher Routes
# -------------------------
@app.route('/teacher/login', methods=["GET", "POST"])
def teacher_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        teacher = User.query.filter_by(username=username, role="teacher").first()
        if teacher and teacher.password == password:
            session["user_id"] = teacher.id
            session["role"] = "teacher"
            return redirect(url_for("teacher_dashboard"))
        else:
            flash("Invalid credentials", "danger")
            return redirect(url_for("teacher_login"))
    return render_template("teacherlogin.html")


@app.route('/teacher/dashboard')
def teacher_dashboard():
    if "user_id" not in session or session.get("role") != "teacher":
        return redirect(url_for("teacher_login"))
    return render_template("teacher_dashboard.html")


@app.route('/teacher/student_profile')
def student_profile():
    if "user_id" not in session or session.get("role") != "teacher":
        return redirect(url_for("teacher_login"))
    students = User.query.filter_by(role="student").all()
    return render_template("student_profile.html", students=students)


@app.route('/teacher/attendance_report')
def attendance_report():
    if "user_id" not in session or session.get("role") != "teacher":
        return redirect(url_for("teacher_login"))
    raw_records = Attendance.query.all()
    records = []
    for rec in raw_records:
        student = User.query.get(rec.student_id)
        records.append({
            'id': rec.id,
            'student_id': rec.student_id,
            'student_name': student.name if student else "Unknown",
            'timestamp': rec.attendance_timestamp,
            'image_path': rec.image_path
        })
    return render_template("attendance_report.html", records=records)


# -------------------------
# Student Routes
# -------------------------
@app.route('/student/login', methods=["GET", "POST"])
def student_login():
    if request.method == "POST":
        student_id = request.form.get("student_id")
        password = request.form.get("password")
        student = User.query.filter_by(username=student_id, role="student").first()
        if student and student.password == password:
            session["user_id"] = student.id
            session["role"] = "student"
            return redirect(url_for("student_dashboard"))
        else:
            flash("Invalid credentials", "danger")
            return redirect(url_for("student_login"))
    return render_template("studentlogin.html")


@app.route('/student/dashboard')
def student_dashboard():
    if "user_id" not in session or session.get("role") != "student":
        return redirect(url_for("student_login"))
    return render_template("student_dashboard.html")


@app.route('/student/attendance')
def student_attendance():
    if "user_id" not in session or session.get("role") != "student":
        return redirect(url_for("student_login"))
    return render_template("student_attendance.html")


@app.route('/student/attendance_history')
def student_attendance_history():
    if "user_id" not in session or session.get("role") != "student":
        return redirect(url_for("student_login"))
    student_id = session.get("user_id")
    raw_records = Attendance.query.filter_by(student_id=student_id).all()
    records = []
    for rec in raw_records:
        records.append({
            'id': rec.id,
            'timestamp': rec.attendance_timestamp,
            'image_path': rec.image_path
        })
    return render_template("student_attendance_history.html", records=records)


# -------------------------
# Preview Recognition (no attendance finalization)
# -------------------------
@app.route('/recognize_preview', methods=["POST"])
def recognize_preview():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    original_filename = secure_filename(file.filename)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    unique_filename = f"{timestamp}_{original_filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(filepath)
    image = Image.open(filepath).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    recognized_class = class_names[predicted.item()]
    # New requirement: if "Kashyap" or "Marmik" is detected, change name to "Monish V"
    if recognized_class in ["Kashyap", "Marmik"]:
        recognized_class = "Monish V"
    return jsonify({"name": recognized_class, "filename": unique_filename})


# -------------------------
# Confirm Attendance (finalize attendance)
# -------------------------
@app.route('/confirm_attendance', methods=["POST"])
def confirm_attendance():
    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "No filename provided"}), 400
    if "user_id" in session and session.get("role") == "student":
        student_id = session["user_id"]
        new_record = Attendance(student_id=student_id, image_path=filename)
        db.session.add(new_record)
        db.session.commit()
        return jsonify({"success": True})
    return jsonify({"error": "Not authorized"}), 403


# -------------------------
# Logout Route
# -------------------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for("index"))


if __name__ == '__main__':
    app.run(debug=True)
