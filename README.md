# COVID-19 Chest X-ray Image Classifier 🧠🩻

This project uses a Convolutional Neural Network (CNN) to classify chest X-ray images into three categories:

- **COVID-19**
- **Normal**
- **Viral Pneumonia**

The model is trained and evaluated on a publicly available dataset of X-ray images.

---

## 📂 Dataset

The dataset used is the [COVID19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database), which includes:

- COVID-19 images
- Normal (healthy) images
- Viral Pneumonia images

Folder structure:

Covid19-dataset/
├── train/
│ ├── Covid/
│ ├── Normal/
│ └── Viral Pneumonia/
└── test/
├── Covid/
├── Normal/
└── Viral Pneumonia/
---

## 🧪 Model Architecture

The CNN architecture includes:

- 3 Convolutional + MaxPooling blocks
- 1 Fully connected Dense layer
- Output layer with softmax for classification

The model uses `categorical_crossentropy` loss and `Adam` optimizer.

---

## 📈 Results

The notebook includes:

- Model training (`model.fit`)
- Accuracy/loss curves
- Sample predictions with actual vs. predicted labels

---

## 🚀 How to Run

1. Clone the repo:
    ```bash
    git clone https://github.com/yourusername/covid19-cnn-classifier.git
    cd covid19-cnn-classifier
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download and place the dataset into `Covid19-dataset/`.

4. Run the notebook:
    ```bash
    jupyter notebook covid19_cnn.ipynb
    ```

---

## 🛠️ Requirements

See [`requirements.txt`](./requirements.txt)

---

## 📌 License

This project is open source under the MIT License.

---

## 🤝 Credits

- Dataset from [Kaggle](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
- Built with TensorFlow & Keras
