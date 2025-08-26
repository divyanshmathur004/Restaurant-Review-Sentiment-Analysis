# Sentiment Analysis on Restaurant Reviews

This project analyzes restaurant reviews and classifies them as either **Positive** or **Negative** using Natural Language Processing (NLP) and various machine learning models. The primary goal is to build an effective classification model and evaluate its performance.

## 📋 Table of Contents

  * [Project Workflow]
  * [Dataset](https://www.kaggle.com/datasets/maher3id/restaurant-reviewstsv)
  * [Technologies Used]
  * [Installation]
  * [Usage]
  * [Model Performance]
  * [Contributing]

## ⚙️ Project Workflow

The project follows a standard machine learning pipeline for NLP tasks:

1.  **Data Loading**: The dataset `Restaurant_Reviews.tsv` is loaded into a Pandas DataFrame.
2.  **Text Preprocessing**: Each review is cleaned by:
      * Removing punctuation and numbers.
      * Converting text to lowercase.
      * Splitting text into individual words (tokenization).
      * Removing common English stop words (e.g., "the", "a", "is").
      * Reducing words to their root form using Porter Stemming (e.g., "loved" becomes "love").
3.  **Feature Extraction**: The cleaned text data is converted into a numerical format using the **Bag of Words** model with `CountVectorizer`. The model is built using the top 1500 most frequent words.
4.  **Model Training**: The data is split into training (80%) and testing (20%) sets. Several classification models are trained:
      * Multinomial Naive Bayes
      * Gaussian Naive Bayes
      * Logistic Regression
      * Random Forest Classifier
5.  **Hyperparameter Tuning**: Each model is tuned to find the best parameters and improve its accuracy.
6.  **Evaluation**: Models are evaluated based on their **Accuracy Score** and a **Confusion Matrix** is generated to visualize performance.

## 📊 Dataset

The dataset used is `Restaurant_Reviews.tsv`, which contains 1000 customer reviews. It has two columns:

  - **Review**: The text of the customer's review.
  - **Liked**: The sentiment label ( `1` for Positive, `0` for Negative).

## 💻 Technologies Used

  * **Python 3**
  * **Jupyter Notebook**
  * **Scikit-learn**: For machine learning models, feature extraction, and metrics.
  * **Pandas**: For data manipulation and loading.
  * **NLTK (Natural Language Toolkit)**: For text preprocessing tasks like stop word removal and stemming.
  * **Matplotlib & Seaborn**: For data visualization and plotting the confusion matrix.
  * **WordCloud**: For visualizing frequent words in the reviews.

## 🚀 Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/YOUR_USERNAME/Restaurant-Review-Sentiment-Analysis.git
    cd Restaurant-Review-Sentiment-Analysis
    ```

2.  **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

    *(Note: You will need to create a `requirements.txt` file containing the libraries listed above.)*

4.  **Download NLTK stopwords:**
    Run Python and enter the following commands:

    ```python
    import nltk
    nltk.download('stopwords')
    ```

## Usage

1.  Place the `Restaurant_Reviews.tsv` file in the root directory of the project.

2.  Launch Jupyter Notebook:

    ```bash
    jupyter notebook
    ```

3.  Open the `Sentiment_Analysis_on_Restaurant_Reviews_IBM.ipynb` file.

4.  Run the cells in order from top to bottom. The final cell allows you to input your own review to get a sentiment prediction.

    **Example Prediction Code:**

    ```python
    # Pass any review string to this function to get a prediction
    def predict_sentiment(sample_review):
        # ... (function definition from the notebook) ...

    # Test with a new review
    my_review = "The food was terrible and the service was slow."

    if predict_sentiment(my_review):
      print("This is a POSITIVE review.")
    else:
      print("This is a NEGATIVE review.")
    ```

## 📈 Model Performance

After training and hyperparameter tuning, the models achieved the following accuracy on the test set. The **Multinomial Naive Bayes** model provided the best performance.

| Model | Best Accuracy Score |
| :--- | :---: |
| **Multinomial Naive Bayes** | **78.5%** |
| Logistic Regression | 74.0% |
| Gaussian Naive Bayes | 73.0% |
| Random Forest Classifier | 71.0% |

The final prediction function uses the tuned Multinomial Naive Bayes model due to its superior performance.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the [issues page](https://www.google.com/search?q=https://github.com/divyanshmathur004/Restaurant-Review-Sentiment-Analysis/).
