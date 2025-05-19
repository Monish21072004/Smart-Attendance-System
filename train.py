import os
import torch
import pickle
import numpy as np
import cv2

from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Paths
DATASET_DIR = r"C:\Users\Monish V\OneDrive\Documents\RANDOM_PROJECTS\Face_detction\Dataset\Enhanced_Images"
MODEL_OUT = r"C:\Users\Monish V\OneDrive\Documents\RANDOM_PROJECTS\Face_detction\Training\face_recog_svm.pkl"

# Setup device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Face detector & embedder
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Gather embeddings and labels
embeddings = []
labels = []

for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_dir): continue

    print(f"→ Processing {person_name}")
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect & crop face
        face = mtcnn(img_rgb)
        if face is None: continue

        # Get embedding
        with torch.no_grad():
            emb = resnet(face.unsqueeze(0).to(device))
        embeddings.append(emb.cpu().numpy()[0])
        labels.append(person_name)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)
X = np.stack(embeddings)

# Train SVM
clf = SVC(kernel='linear', probability=True)
clf.fit(X, y)

# Save SVM + label encoder
with open(MODEL_OUT, 'wb') as f:
    pickle.dump({'svm': clf, 'le': le}, f)

print(f"Trained on {len(y)} samples, saved model to {MODEL_OUT}")
