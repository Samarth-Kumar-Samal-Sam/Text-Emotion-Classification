---

# 😃 Text Emotion Classification Web Application

### 🚀 **Live Demo**

🔗 [Try the Application Here](https://sam-text-emotion.streamlit.app/) — **Experience real-time emotion detection powered by NLP and machine learning!**

---

## 📌 Project Overview

Understand how people feel just from their words using this smart **Text Emotion Classification** app! Built with Natural Language Processing (NLP) techniques, this tool classifies emotions like *joy*, *anger*, *fear*, and more from user-provided text.

Developed using **Streamlit** for an intuitive interface, it allows users to input any sentence and get instant emotion predictions using a pre-trained machine learning model. Whether you're analyzing tweets, enhancing chatbots, or conducting sentiment research, this app is built for you!

---

## 🛠️ Tech Stack & Tools

| Technology              | Purpose                                 |
| ----------------------- | --------------------------------------- |
| 🐍 Python 3.8+          | Core programming language               |
| 🚀 Streamlit            | Interactive web interface               |
| 🧠 Scikit-learn         | Machine learning model development      |
| 🗣️ NLTK                | Text preprocessing and NLP operations   |
| 📦 Joblib               | Model serialization and loading         |
| 📊 Matplotlib & Seaborn | Data visualization and analysis         |
| 📈 Altair               | Interactive charting                    |
| 🐼 Pandas & NumPy       | Data processing and numerical computing |

---

## ✨ Key Features

* 📝 **Real-Time Emotion Detection**: Classify user input text into emotions instantly
* 🎨 **Dynamic Visualizations**: See emotion distribution through interactive charts
* 🔍 **Multi-Emotion Support**: Classifies text into categories like Joy, Sadness, Anger, and more
* 📂 **Structured Workflow**: Modular design with separate folders for models, datasets, and notebooks
* 📥 **Trained Model Integration**: Easily extend or replace the model using the training notebook
* 📱 **Responsive Design**: Works across desktop, tablet, and mobile screens

---

## ⚙️ Setup Instructions (Local Development)

### 1. Clone the Repository

```bash
git clone https://github.com/Samarth-Kumar-Samal/Emotion-Text-Classification-using-NLP.git

cd Emotion-Text-Classification-using-NLP
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

* **Windows:**

```bash
venv\Scripts\activate
```

* **Linux/macOS:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```plaintext
.
├── Assets/
│   └── Text.jpg                     # Image used in the web app
├── Dataset/
│   ├── Emotion Dataset.csv          # Raw labeled data
│   ├── training_dataset.csv         # Cleaned training data
│   └── testing_dataset.csv          # Cleaned testing data
├── Model/
│   └── Text-Model.joblib            # Trained model file
├── Notebooks/
│   ├── Training Notebook.ipynb      # Model training workflow
│   └── Testing Notebook.ipynb       # Model evaluation and analysis
├── .gitignore                       # Git ignore file
├── LICENSE                          # License file
├── README.md                        # Project documentation
├── app.py                           # Streamlit application
└── requirements.txt                 # Python dependencies
```

---

## 🚀 Usage Instructions

1. Launch the app via Streamlit.
2. Enter a sentence or paragraph into the input field.
3. Click **Predict Emotion** to get the classified emotion label.
4. Explore the **emotion distribution** chart for visual context.
5. Optionally, inspect and retrain the model using the included notebooks.

---

## 🧪 Model Training & Evaluation

Use the provided notebooks to build or refine the ML model:

* **Training Notebook.ipynb**:

  * Text preprocessing with NLTK
  * Vectorization using `CountVectorizer` or `TfidfVectorizer`
  * Train a classifier like Logistic Regression or SVM
  * Export model using `joblib`

* **Testing Notebook.ipynb**:

  * Load and test model performance
  * Visualize confusion matrix
  * Generate classification report (accuracy, precision, recall, F1-score)

---

## 📊 Emotion Categories

The classifier currently detects the following emotions:

* 😄 **Joy**
* 😡 **Anger**
* 😢 **Sadness**
* 😨 **Fear**
* 😲 **Surprise**
* 😐 **Neutral**
* 😍 **Love**

---

## 👨‍💻 Contributing

We welcome your contributions! Follow these steps:

1. Fork the repository

2. Create a feature branch:

   ```bash
   git checkout -b feature-name
   ```

3. Commit your changes:

   ```bash
   git commit -m "Add feature description"
   ```

4. Push to your branch:

   ```bash
   git push origin feature-name
   ```

5. Submit a Pull Request 🚀

---

## 📜 License

This project is licensed under the **[MIT License](LICENSE)**.

---

## 👤 Author

**Samarth Kumar Samal**
🔗 [GitHub Profile](https://github.com/Samarth-Kumar-Samal-Sam)

---

## 🙏 Acknowledgements

Special thanks to these powerful libraries and tools:

* [Scikit-learn](https://scikit-learn.org/)
* [NLTK](https://www.nltk.org/)
* [Pandas](https://pandas.pydata.org/)
* [NumPy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [Altair](https://altair-viz.github.io/)
* [Streamlit](https://streamlit.io/)
---
