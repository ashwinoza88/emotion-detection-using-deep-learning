
# Emotion Detection Using Deep Learning

This project demonstrates facial emotion detection using Deep Learning with the FER-2013 dataset. It compares two models: a custom-built CNN for three-class classification and a fine-tuned VGG16 model for seven-class classification. The models were developed by a team of three as part of a deep learning course project at VIT-AP University.

---

## 📌 Table of Contents
- [Project Brief](#project-brief)
- [Dataset](#dataset)
- [Models & Techniques](#models--techniques)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)
- [License](#license)

---

## 📖 Project Brief

This project focuses on facial emotion recognition using grayscale images. Two deep learning models were developed:

1. **Custom DCNN Model** (by Ashwin Oza)  
   - Focused on **3 classes**: `Happy`, `Sad`, and `Neutral`.
   - Utilized a custom CNN architecture with ELU activations, dropout regularization, and batch normalization.
   - Achieved **82.30%** validation accuracy.

2. **Fine-tuned VGG16 Model** (by Musaddiq Ajaz)  
   - Used all **7 emotion classes** from FER-2013.
   - Employed transfer learning using a pre-trained VGG16 model.
   - Achieved **67.93%** validation accuracy.

---

## 📂 Dataset

- **FER-2013 Dataset** (available on [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013))
- Contains 35,887 grayscale 48×48 facial expression images categorized into 7 emotions:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral

---

## 🧠 Models & Techniques

### 🔹 Custom DCNN (3-Class)
- ELU activations, Batch Normalization, Dropout.
- 3 Conv blocks followed by dense layers.
- Loss: Categorical Crossentropy  
- Optimizer: Adam

### 🔹 Fine-Tuned VGG16 (7-Class)
- Used VGG16 base (from ImageNet), added Global Average Pooling and Dense layers.
- Entire model was fine-tuned.
- Loss: Categorical Crossentropy  
- Optimizer: Adam

---

## 📊 Results

| Model | Classes | Accuracy |
|-------|---------|----------|
| DCNN  | 3       | 82.30%   |
| VGG16 | 7       | 67.93%   |

- Confusion matrices and output predictions are included in the notebooks.
- Performance graphs visualize training and validation metrics.

---

## ⚙️ Installation

```bash
git clone https://github.com/ashwinoza88/emotion-detection-using-deep-learning.git
cd emotion-detection-using-deep-learning
```

---

## ▶️ Usage

This project consists of two Jupyter Notebooks built and executed using Kaggle Notebook. You can easily run them online using Kaggle's GPU-accelerated environment, or download and open them locally using Jupyter Notebook.

### 📁 Notebooks

- `FER-2013-DCNN.ipynb`  
  Custom Deep Convolutional Neural Network (DCNN) for **3-class emotion detection** (`Happy`, `Sad`, `Neutral`). This model focuses on simplicity and accuracy, achieving ~82.30% validation accuracy.

- `FER-2013-VGG16(i).ipynb`  
  Fine-tuned pre-trained **VGG16** model for **7-class classification** using the full FER-2013 dataset (`Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`). This model leverages transfer learning and deeper feature representations.

### ✅ Run on Kaggle (Recommended)

1. Visit [Kaggle - FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013) and add it to your Kaggle account.
2. Upload the notebooks to your Kaggle account.
3. Set GPU as the accelerator in "Notebook Settings" → `Change Runtime Type` → GPU.
4. Run each cell sequentially to train and evaluate the models.

### 🖥️ Run Locally (Optional)

If you prefer running the notebooks locally:

1. Download the `.ipynb` files.
2. Install [Jupyter Notebook](https://jupyter.org/install) and required libraries (`tensorflow`, `keras`, `numpy`, `pandas`, `matplotlib`).
3. Download the FER-2013 dataset and place it in the expected path (as referenced in the code).
4. Open a terminal:

```bash
jupyter notebook
```

Then open the desired notebook and execute the cells.

📌 **Note:** This project was originally developed and tested entirely on **Kaggle** using their built-in Python environment and GPU backend.

---

## 👨‍💻 Contributors

This project was a team effort by 3 undergraduate students from the School of Computer Science and Engineering, VIT-AP University, Amaravati.

- **Oza Ashwin Kumar Ramanbhai**  
  - Roll No: 22BCE7932  
  - Built and trained the **custom CNN (DCNN)** model for 3 emotion classes  
  - GitHub: [@ashwinoza88](https://github.com/ashwinoza88)  
  - Email: ashwinkumar.22bce7932@vitapstudent.ac.in

- **Manav Mehta**  
  - Roll No: 22BCE7785  
  - Assisted with data preprocessing and report writing  
  - GitHub: [@manava10](https://github.com/manava10)  
  - Email: manav.22bce7785@vitapstudent.ac.in

- **M. Musaddiq Ajaz**  
  - Roll No: 22BCE9253  
  - Implemented and trained the **fine-tuned VGG16** model for all 7 emotions  
  - GitHub: [@musaddiq82](https://github.com/musaddiq82)  
  - Email: ajaz.22bce9253@vitapstudent.ac.in

Feel free to connect with us for collaboration or feedback!

---

## 📝 License

This project is licensed under the MIT License.  
You are free to use, modify, and share this work with proper attribution.

See the [LICENSE](LICENSE) file for more details.
