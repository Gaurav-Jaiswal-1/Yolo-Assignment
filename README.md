# 🧴 Bottle Quality Detection System (YOLOv8 Classification)

A deep learning project that classifies plastic bottles as **GOOD** or **BAD** using a YOLOv8 classification model.
The system includes training, evaluation, batch prediction, real-time detection, and a Streamlit web app.

---

## 🚀 Features

✅ Image classification using YOLOv8
✅ Predict single image or folder
✅ Real-time webcam classification
✅ Streamlit web interface
✅ Confidence score display
✅ Model comparison (augmented vs non-augmented)
✅ Lightweight and fast inference

---

## 🧠 Problem Statement

Automatically detect whether a plastic bottle is in **GOOD condition** or **DAMAGED/BAD condition** using computer vision.

---

## 🏗 Project Structure

```
project/
│
├── app.py                     # Streamlit web app (classification UI)
├── realtime.py                # Real-time webcam script
├── predict_folder.py          # Batch prediction script
│
├── models/
│   ├── classifier/
│   │     └── best.pt          # Trained classification model
│   ├── aug_model/
│   │     └── best.pt          # Augmented model
│   └── yolov8n/
│         └── best.pt          # Non-augmented model
│
├── dataset/
│   ├── train/
│   │    ├── good/
│   │    └── bad/
│   ├── val/
│   │    ├── good/
│   │    └── bad/
│   └── test/
│        ├── good/
│        └── bad/
│
├── runs/
│   └── classify/              # Training logs and results
│
├── requirements.txt
└── README.md
```

---

## 🧪 Dataset Format (Classification)

Images are organized into folders where folder names act as labels.

```
dataset/
   train/
      good/
      bad/
   val/
      good/
      bad/
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```
git clone https://github.com/your-username/bottle-quality-yolo.git
cd bottle-quality-yolo
```

### 2️⃣ Create Virtual Environment

```
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## 📦 Requirements

```
ultralytics
streamlit
opencv-python
pillow
numpy
```

---

## 🏋️ Training Model

```
from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="dataset",
    epochs=50,
    imgsz=224,
    batch=8
)
```

Trained weights will be saved in:

```
runs/classify/train/weights/best.pt
```

---

## 🔍 Predict Single Image

```
from ultralytics import YOLO

model = YOLO("models/classifier/best.pt")

results = model("test.jpg")

probs = results[0].probs
label = results[0].names[probs.top1]
conf = probs.top1conf

print(label, conf)
```

---

## 📂 Predict Folder

```
python predict_folder.py
```

Outputs predictions with confidence scores.

---

## 🎥 Real-Time Detection

```
python realtime.py
```

Press **Q** to exit webcam.

---

## 🌐 Run Streamlit App

```
streamlit run app.py
```

Open browser at:

```
http://localhost:8501
```

---

## 📊 Model Output

The model predicts:

* Bottle condition (GOOD / BAD)
* Confidence score
* Class probabilities

---

## 🧠 Model Type

YOLOv8 Classification (yolov8n-cls)

Why classification?

✔ Faster
✔ Needs less data
✔ Simpler pipeline
✔ Better for binary quality inspection

---

## 📈 Evaluation Metrics

* Accuracy
* Confusion matrix
* Confidence score

Training logs saved in:

```
runs/classify/
```

---

## 🏭 Real-World Applications

* Industrial quality inspection
* Manufacturing automation
* Defect detection
* Smart sorting systems

---

## ⚠️ Limitations

* Small dataset may cause overfitting
* Performance depends on lighting conditions
* Needs more bad samples for better generalization

---

## 🚀 Future Improvements

⭐ Collect more real defect images
⭐ Add alert system for BAD detection
⭐ Deploy as REST API
⭐ Improve UI dashboard
⭐ Add model monitoring

---

## 👨‍💻 Author

Gaurav Jaiswal

---

## 📜 License

This project is for educational and research purposes.

---

## ⭐ Acknowledgements

* Ultralytics YOLOv8
* OpenCV
* Streamlit
* PyTorch

---

If you found this project useful, consider giving it a ⭐
