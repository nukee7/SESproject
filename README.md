Marketsense overview-
This Project is a stock market sentiment analysis and prediction system built using Python and Streamlit. It uses the Polygon API to fetch real-time stock data and news articles. Sentiment analysis is performed with the FinBERT model, while stock price prediction is done using a hybrid model combining LSTM and XGBoost. The app provides an interactive dashboard where users can input stock symbols, view sentiment scores, and see future price predictions based on sentiment and technical indicators.
To run the project, clone the repository, install the required packages from requirements.txt, and launch the Streamlit app with streamlit run app.py. This project is designed to demonstrate how machine learning, sentiment analysis, and financial data can be combined into an easy-to-use web application. It is ideal for anyone interested in finance, AI, or real-time data-driven stock market predictions.

Team members information:
Nikhil kumar-23bds038@iiitdwd.ac.in
Palak gupta-23bds042@iiitdwd.ac.in
Pradnesh Fernandez-23bds044@iiitdwd.ac.in
Shivansh shukla-23bds054@iiitdwd.ac.in
Shree vats-23bds055@iiitdwd.ac.in
Vaibhav sharma- 23bds066@iiitdwd.ac.in

How to use this git repo-
Installation
1. Clone the Repository


git clone https://github.com/nukee7/SESproject.git



Install the Required Packages

Make sure you have Python 3.11 or above.


pip install -r requirements.txt


3. Set Up API Keys

If you are using APIs like Polygon API or Google News API, create a `.env` file:


POLYGON_API_KEY=your_polygon_api_key
NEWS_API_KEY=your_news_api_key


Or directly add keys inside the code.

## 🛠 Usage

Run the main script:

```bash
python se.py
```

Or if it’s a Streamlit app:

```bash
streamlit run se.py
```

✅ It will start analyzing and predicting based on live stock data and news sentiment.


## 📚 Project Structure

```bash
├── se.py                # Main project script
├── requirements.txt     # List of dependencies
├── README.md            # Project instructions (this file)
├── .env                 # (Optional) API keys
└── umldiagrams
```

---

 Tech Stack

Python
Tensorflow - Deep Learning (BERT Sentiment Model)
XGBoost - Stock Price Regression
Streamlit - Frontend UI
Polygon API - Stock data
ta-lib / ta - Technical Indicators
