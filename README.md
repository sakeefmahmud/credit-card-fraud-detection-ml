# Credit Card Fraud Detection

This project builds a **machine learning model** to detect fraudulent credit card transactions using the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). It uses **Random Forest Classifier** to distinguish between valid and fraudulent transactions.

The workflow was developed in **Google Colab** and saves the trained model as a `.pkl` file for reuse.

---

## 📌 Project Overview

* Dataset: **284,807 transactions, 31 features**
* Highly imbalanced dataset:

  * Normal transactions → 284,315
  * Fraudulent transactions → 492
* To handle imbalance, the dataset was **undersampled** (equalizing fraud and non-fraud cases).
* Trained with **Random Forest Classifier**.
* Achieved **~95% accuracy** on the test set.
* Model is saved using **Joblib** for deployment and prediction.

---

## 🛠️ Tech Stack

* **Python**
* **Pandas, NumPy** (Data handling)
* **Scikit-learn** (Model building, training, evaluation)
* **Joblib** (Model saving & loading)
* **Google Colab** (Development environment)

---

## ⚙️ How It Works

1. **Mount Google Drive & Load Dataset**

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   file = '/content/drive/My Drive/Colab Notebooks/Datasets/creditcard.csv'
   ```

2. **Data Preprocessing**

   * Removed class imbalance by sampling equal normal and fraud cases.
   * Split dataset into training and testing sets.

3. **Model Training**

   ```python
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score

   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
   model = RandomForestClassifier()
   model.fit(x_train, y_train)
   ```

4. **Model Evaluation**

   ```python
   predictions = model.predict(x_test)
   score = accuracy_score(predictions, y_test)
   print(f"Accuracy Score: {score}")
   ```

5. **Save and Reload Model**

   ```python
   import joblib
   joblib.dump(model, 'creditcard.pkl')
   model = joblib.load('creditcard.pkl')
   ```

6. **Make Predictions**

   ```python
   new_input = np.ones((1, 30))  # Example input
   prediction = model.predict(new_input)
   if prediction[0] == 0:
       print("Not Fraud")
   else:
       print("Fraud")
   ```

---

## 📊 Results

* **Accuracy:** ~95%
* Balanced dataset of **984 samples** (492 fraud, 492 normal).

---

## 🚀 Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Open in Google Colab or Jupyter Notebook.

3. Run all cells to:

   * Train the model
   * Evaluate performance
   * Save model as `creditcard.pkl`
   * Test predictions with new input

---

## 📂 Repository Structure

```
credit-card-fraud-detection/
│── creditcard.csv              # Dataset (not included, download from Kaggle)
│── creditcard.pkl              # Trained model (saved with Joblib)
│── fraud_detection.ipynb       # Colab Notebook code
│── README.md                   # Project documentation
```


## 🙌 Acknowledgments

Dataset: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---
