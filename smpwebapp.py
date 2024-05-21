'''

from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf
import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_symbol = request.form['stock_symbol']
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365)
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    data = data[['Close']]
    data['Next Close'] = data['Close'].shift(-1)
    data = data.dropna()

    X = data[['Close']]
    y = data['Next Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    future_date = end_date + datetime.timedelta(days=1)
    predicted_price = model.predict([[data.iloc[-1]['Close']]])[0]

    return render_template('result.html', stock_symbol=stock_symbol, predicted_price=predicted_price, future_date=future_date)

if __name__ == '__main__':
    app.run(debug=True)

'''



from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        stock_symbol = request.form['stock_symbol']
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=365)
        data = yf.download(stock_symbol, start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError("Error: No data available for the provided stock symbol.")

        data = data[['Close']]
        data['Next Close'] = data['Close'].shift(-1)
        data = data.dropna()

        X = data[['Close']]
        y = data['Next Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        future_date = end_date + datetime.timedelta(days=1)
        predicted_price = model.predict([[data.iloc[-1]['Close']]])[0]

        # Generate a simple line chart
        plt.figure(figsize=(8, 4))
        plt.plot(data['Close'], label='Actual Close Price')
        plt.scatter(data.index[-1], data['Close'].iloc[-1], color='red')  # Scatter plot for last actual value
        plt.plot(data.index, model.predict(X), label='Predicted Close Price', linestyle='--')
        plt.scatter(future_date, predicted_price, color='blue')  # Scatter plot for predicted value
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title('Stock Price Prediction')
        plt.legend()

        # Save the plot to a BytesIO object
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_data = base64.b64encode(img_buf.read()).decode('utf-8')

        # Close the plot to free up resources
        plt.close()

        return render_template('result.html', stock_symbol=stock_symbol, predicted_price=predicted_price,
                               future_date=future_date, graph=img_data)

    except Exception as e:
        error_message = f"Error: {str(e)}"
        return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)




 

