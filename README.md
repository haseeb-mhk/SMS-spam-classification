# SMS Spam Classification Project

## Overview
This project demonstrates the implementation of a Naive Bayes classifier for SMS spam detection. The objective is to classify text messages as either **spam** or **ham** (not spam) using Natural Language Processing (NLP) and machine learning techniques.

---

## Dataset
The project utilizes the **SMS Spam Collection Dataset**, which contains labeled messages in two categories:
- **ham**: Legitimate messages
- **spam**: Unsolicited or junk messages

The dataset is processed to ensure proper formatting and cleanliness before training the model.

---

## Code Details

### **1. Data Preprocessing**
- **Loading Data:** The dataset is read from a CSV file.
- **Data Cleaning:**
  - Checked for and handled null values.
  - Renamed columns for clarity: `status` (labels) and `message` (text).
  - Mapped `ham` to `0` and `spam` to `1` for binary classification.
- **Splitting Data:** The dataset was divided into training (80%) and testing (20%) sets using `train_test_split`.

### **2. Text Transformation**
- Used the **Bag of Words** model with `CountVectorizer` to convert text data into numerical format.
- The training data was fitted and transformed, and the testing data was transformed using the same vectorizer.

### **3. Model Training**
- A **Multinomial Naive Bayes** classifier was trained on the transformed training data.

### **4. Model Evaluation**
- Predicted labels for the test set.
- Evaluated performance using:
  - **Accuracy**
  - **Classification Report** (Precision, Recall, F1-Score)
  - **Confusion Matrix**

### **5. Custom Message Testing**
- The model was tested on custom messages to demonstrate its predictive capabilities.
- Example:
  - Message: "Free entry in a weekly competition!" -> **Spam**
  - Message: "Hey, are you free tomorrow?" -> **Ham**

---

## Results
- **Accuracy:** Achieved an accuracy of approximately **[insert accuracy]%** on the test set.
- The confusion matrix and classification report further confirmed the model's reliability in distinguishing between spam and ham messages.

---

## Tools and Technologies
- **Programming Language:** Python
- **Libraries Used:**
  - Pandas and NumPy for data handling and manipulation
  - Scikit-learn for machine learning and evaluation metrics
  - CountVectorizer for feature extraction
- **IDE:** Jupyter Notebook

---

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd sms-spam-classification
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook sms_spam_classification.ipynb
   ```
4. Test the classifier by editing the `messages` list with your custom SMS texts.

---

## Challenges and Learnings
- Handling and preprocessing textual data for machine learning.
- Understanding the trade-offs between different evaluation metrics.
- Implementing and evaluating a Naive Bayes algorithm for a real-world problem.

---

## Future Work
- Incorporate advanced NLP techniques such as **TF-IDF** or deep learning models like **LSTM** for improved accuracy.
- Develop a simple web application for real-time SMS spam detection.

---

## Sample Output
### Example Predictions
| Message                                      | Prediction |
|----------------------------------------------|------------|
| Free entry in a weekly competition!          | Spam       |
| Hey, are you free tomorrow?                  | Ham        |

---
