## Smart Attendance System

A Python-based face-recognition attendance solution featuring both an offline training pipeline and a Flask web application for live camera streaming and image-upload attendance.

---

## Table of Contents

* [Features](#features)
* [Technologies](#technologies)
* [Getting Started](#getting-started)

  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Training the Model](#training-the-model)
* [Configuration](#configuration)
* [Running the Application](#running-the-application)
* [Project Structure](#project-structure)
* [Usage](#usage)

  * [Teacher Interface](#teacher-interface)
  * [Student Interface](#student-interface)
* [Folder Structure](#folder-structure)
* [Contributing](#contributing)
* [License](#license)

---

## Features

* **Face Detection**: Uses MTCNN (facenet-pytorch) for accurate face detection and cropping.
* **Embeddings & Classification**: Generates 512-D face embeddings with InceptionResnetV1 and classifies via an SVM model.
* **Live Attendance**: Real-time attendance capture through webcam streaming (OpenCV + Haar Cascades).
* **Image Upload**: Supports image-upload attendance with preview and confirmation.
* **User Roles**: Separate teacher and student dashboards with role-based access.
* **Database-backed**: Stores users and attendance logs in MySQL via SQLAlchemy.

---

## Technologies

* Python 3.8+
* Flask
* facenet-pytorch (MTCNN + InceptionResnetV1)
* scikit-learn (SVM)
* OpenCV
* SQLAlchemy
* MySQL
* PyTorch

---

## Getting Started

### Prerequisites

* Python 3.8 or higher
* MySQL database
* `virtualenv` (recommended)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Monish21072004/Smart-Attendance-System.git
   cd Smart-Attendance-System
   ```
2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Training the Model

1. Organize your dataset under `DATASET_DIR`, with one subfolder per person containing their images.
2. Run the training script:

   ```bash
   python train.py
   ```
3. This will generate `face_recog_svm.pkl`, which contains the trained SVM model and label encoder.

---

## Configuration

Edit the following in `app.py` before running:

* `UPLOAD_FOLDER` – Directory to store uploaded images
* `SQLALCHEMY_DATABASE_URI` – Your MySQL connection string
* `SECRET_KEY` – Flask session secret

---

## Running the Application

Start the Flask server:

```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000/`.

---

## Project Structure

```
Smart-Attendance-System/
├── train.py                   # Training pipeline script
├── app.py                     # Flask web application
├── face_recog_svm.pkl         # Serialized model & label encoder
├── requirements.txt           # Python dependencies
├── haarcascade_frontalface_default.xml  # Haar Cascade for OpenCV
├── templates/                 # HTML templates
│   ├── login.html
│   ├── teacher_dashboard.html
│   ├── student_dashboard.html
│   ├── live.html
│   └── ...
├── static/                    # CSS, JS, images
├── DATASET_DIR/               # Training images (per-person folders)
└── uploads/                   # Stored uploads and snapshots
```

---

## Usage

### Teacher Interface

1. Navigate to `/teacher/login`
2. View all students and their attendance records
3. Download or export attendance reports

### Student Interface

1. Navigate to `/student/login`
2. Choose `Live` or `Upload` mode for attendance
3. Preview recognition result
4. Confirm attendance to log it in the database

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/MyFeature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/MyFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the [GPL-3.0 License](LICENSE).
