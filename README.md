# 📈 MarketSense

**MarketSense** is a stock market sentiment analysis and prediction system built using Python and Streamlit.  
It combines real-time stock data, news sentiment analysis, and machine learning models to provide stock price predictions.

---

## 🧠 Project Overview

- Fetches real-time stock data and news articles using **Polygon API**.
- Performs sentiment analysis using the **FinBERT** model.
- Predicts stock prices with a **hybrid LSTM + XGBoost** model.
- Provides an interactive dashboard where users:
  - Input stock symbols
  - View sentiment scores
  - See future price predictions based on sentiment and technical indicators

This project demonstrates how **machine learning**, **sentiment analysis**, and **financial data** can be combined into a real-time, user-friendly application.  
Ideal for finance enthusiasts, AI researchers, and anyone interested in stock market predictions!

---

## 👥 Team Members

- **Nikhil Kumar** - 23bds038@iiitdwd.ac.in
- **Palak Gupta** - 23bds042@iiitdwd.ac.in
- **Pradnesh Fernandez** - 23bds044@iiitdwd.ac.in
- **Shivansh Shukla** - 23bds054@iiitdwd.ac.in
- **Shree Vats** - 23bds055@iiitdwd.ac.in
- **Vaibhav Sharma** - 23bds066@iiitdwd.ac.in

---

## 🚀 How to Use This Git Repository

### 1. Clone the Repository

```bash
git clone https://github.com/nukee7/SESproject.git
cd SESproject
```

### 2. Install the Required Packages

Make sure you have **Python 3.11** or above installed.

```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys

Create a `.env` file in the root directory:

```plaintext
POLYGON_API_KEY=your_polygon_api_key
NEWS_API_KEY=your_news_api_key
```

Alternatively, you can directly add your API keys inside the code.

---

## 🛠️ Usage

### Run the main script:

```bash
python se.py
```

### Or if it’s a Streamlit app:

```bash
streamlit run se.py
```

✅ This will start analyzing and predicting based on live stock data and news sentiment.

---

## 📂 Project Structure

```bash
├── se.py                # Main project script
├── requirements.txt     # List of dependencies
├── README.md            # Project instructions (this file)
├── .env                 # (Optional) API keys
└── umldiagrams/         # UML Diagrams for the project
```

---

## 🛠 Tech Stack

- **Python**
- **TensorFlow** - Deep Learning (BERT Sentiment Model)
- **XGBoost** - Stock Price Regression
- **Streamlit** - Frontend UI
- **Polygon API** - Stock data
- **ta-lib / ta** - Technical Indicators

