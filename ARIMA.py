# Importar as bibliotecas necessárias
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
from sklearn.metrics import mean_absolute_percentage_error
import json

# Ler os dados
file_path = 'merge_dataframes.csv'
data = pd.read_csv(file_path)

# data = data.iloc[6150:]
data = data.iloc[100:]

# Ordenar os dados por data
data = data.sort_values(by='datetime')

# Converter a coluna 'datetime' para datetime
data['datetime'] = pd.to_datetime(data['datetime'])

# Função para calcular RSI
def calculate_rsi(df, window=14):
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calcular RSI
data['rsi'] = calculate_rsi(data).ffill()

# Calcular SMA (50 e 200 períodos)
data['SMA_50'] = data['close'].rolling(window=50).mean()
data['SMA_200'] = data['close'].rolling(window=200).mean()

# Calcular Bollinger Bands
data['BB_middle'] = data['close'].rolling(window=20).mean()
data['BB_std'] = data['close'].rolling(window=20).std()
data['BB_upper'] = data['BB_middle'] + (data['BB_std'] * 2)
data['BB_lower'] = data['BB_middle'] - (data['BB_std'] * 2)

# Calcular MACD (EMA 12 e 26 períodos)
data['EMA_12'] = data['close'].ewm(span=12, adjust=False).mean()
data['EMA_26'] = data['close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA_12'] - data['EMA_26']
data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
data['MACD_histogram'] = data['MACD'] - data['MACD_signal']

# Separar os valores mais recentes para usar no JSON
latest_rsi = data['rsi'].iloc[-1]
latest_sma_50 = data['SMA_50'].iloc[-1]
latest_sma_200 = data['SMA_200'].iloc[-1]
latest_bb_upper = data['BB_upper'].iloc[-1]
latest_bb_lower = data['BB_lower'].iloc[-1]
latest_macd = data['MACD'].iloc[-1]
latest_macd_signal = data['MACD_signal'].iloc[-1]
latest_macd_histogram = data['MACD_histogram'].iloc[-1]

# Definir o sinal MACD
if latest_macd < latest_macd_signal:
    macd_signal = "sell_signal" if latest_macd_histogram < 0 else "potential_reversal"
else:
    macd_signal = "buy_signal"

# Definir probabilidades dinâmicas com base nos indicadores técnicos
probabilities = {
    "sell_off": 0.7 if latest_rsi > 70 and macd_signal == "sell_signal" else 0.2,
    "pullback": 0.6 if latest_rsi < 30 and macd_signal == "buy_signal" else 0.25,
    "bear_trap": 0.5 if latest_rsi < 30 and latest_macd > latest_macd_signal else 0.1
}

# Definir ponto de apoio para stop loss e sell loss usando as bandas de Bollinger
stop_loss = latest_bb_lower  # Usar a banda inferior como stop loss
sell_loss = latest_bb_upper  # Usar a banda superior como ponto de sell limit (take profit)

# Preparar os dados exógenos e as previsões com ARIMAX
exogenous_data = data[['open', 'high', 'low', 'ticker_sentiment_score']].values
close_prices = data['close'].values

# Ajustar o modelo ARIMAX com variáveis exógenas adicionais
model_arimax = ARIMA(close_prices, order=(5, 1, 0), exog=exogenous_data)
model_arimax_fit = model_arimax.fit()

# Dividir os dados em treinamento (80%) e teste (20%) para validação
train_size = int(len(data) * 0.8)  # 80% para treinamento
train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]

# Extrair dados exógenos e os preços de fechamento para treino e teste
exogenous_train = train_data[['open', 'high', 'low', 'ticker_sentiment_score']].values
exogenous_test = test_data[['open', 'high', 'low', 'ticker_sentiment_score']].values
close_prices_train = train_data['close'].values
close_prices_test = test_data['close'].values

# Prever os preços no conjunto de teste
forecast_test = model_arimax_fit.forecast(steps=len(close_prices_test), exog=exogenous_test)

# Calcular MAPE (Mean Absolute Percentage Error) para ver a precisão do modelo
mape = mean_absolute_percentage_error(close_prices_test, forecast_test)

# Definir as variáveis exógenas para 5, 10, 15 e 60 minutos corretamente
next_5_min_exog = np.array([exogenous_data[-1]] * 5)
next_10_min_exog = np.array([exogenous_data[-1]] * 10)
next_15_min_exog = np.array([exogenous_data[-1]] * 15)
next_60_min_exog = np.array([exogenous_data[-1]] * 60)

# Previsões de preço
forecast_5_min = model_arimax_fit.forecast(steps=5, exog=next_5_min_exog)[-1]
forecast_10_min = model_arimax_fit.forecast(steps=10, exog=next_10_min_exog)[-1]
forecast_15_min = model_arimax_fit.forecast(steps=15, exog=next_15_min_exog)[-1]
forecast_60_min = model_arimax_fit.forecast(steps=60, exog=next_60_min_exog)[-1]

# Última data para basear as previsões
last_datetime = data['datetime'].iloc[-1]

# Adicionar as datas futuras às previsões (assumindo intervalos de 1 minuto)
forecast_5_min_time = last_datetime + timedelta(minutes=5)
forecast_10_min_time = last_datetime + timedelta(minutes=10)
forecast_15_min_time = last_datetime + timedelta(minutes=15)
forecast_60_min_time = last_datetime + timedelta(minutes=60)

# Obter os últimos 100 dados do CSV e converter a coluna datetime para string
last_100_data = data.tail(500).copy()
# only datetime open high low close
last_100_data = last_100_data[['datetime', 'open', 'high', 'low', 'close']]
last_100_data['datetime'] = last_100_data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
last_100_data_dict = last_100_data.to_dict(orient='records')
# convert to array
last_100_data = []
for i in last_100_data_dict:
    last_100_data.append(i)


# Construir o JSON com as previsões em formato de array
output_json_real_indicators = {
    "price_forecast": [
        {
            "prevision_time_min": 5,
            "predicted_price": forecast_5_min,
            "datetime": forecast_5_min_time.strftime('%Y-%m-%d %H:%M:%S'),
            "confidence_interval": [forecast_5_min - 0.1, forecast_5_min + 0.1]
        },
        {
            "prevision_time_min": 10,
            "predicted_price": forecast_10_min,
            "datetime": forecast_10_min_time.strftime('%Y-%m-%d %H:%M:%S'),
            "confidence_interval": [forecast_10_min - 0.1, forecast_10_min + 0.1]
        },
        {
            "prevision_time_min": 15,
            "predicted_price": forecast_15_min,
            "datetime": forecast_15_min_time.strftime('%Y-%m-%d %H:%M:%S'),
            "confidence_interval": [forecast_15_min - 0.1, forecast_15_min + 0.1]
        },
        {
            "prevision_time_min": 60,
            "predicted_price": forecast_60_min,
            "datetime": forecast_60_min_time.strftime('%Y-%m-%d %H:%M:%S'),
            "confidence_interval": [forecast_60_min - 0.1, forecast_60_min + 0.1]
        }
    ],
    "technical_indicators": {
        "rsi": latest_rsi,
        "sma_50": latest_sma_50,
        "sma_200": latest_sma_200,
        "bollinger_bands": {
            "upper_band": latest_bb_upper,
            "lower_band": latest_bb_lower
        },
        "macd": {
            "value": latest_macd,
            "signal": latest_macd_signal,
            "histogram": latest_macd_histogram,
            "signal_strength": macd_signal
        }
    },
    "probabilities": probabilities,
    "stop_loss": stop_loss,
    "mape": mape,
    "ticker_sentiment_score": exogenous_data[-1][-1],

}

print(json.dumps(output_json_real_indicators, indent=4))
