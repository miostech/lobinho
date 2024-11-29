import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import json

# Carregar os dados
data = pd.read_json('merge_dataframes.json', lines=True)

# Preparar os dados de fechamento
close_prices = data['close'].values

# Normalizar os dados
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(close_prices).reshape(-1, 1))

# Dividir os dados em treino e teste (80% para treino)
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Função para criar o conjunto de dados para LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Definir a quantidade de passos de tempo (ex: 60 minutos)
time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Redimensionar os dados para [samples, time_steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Criar o modelo LSTM
model_lstm = Sequential()
model_lstm.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model_lstm.add(LSTM(50, return_sequences=False))
model_lstm.add(Dense(25))
model_lstm.add(Dense(1))

# Compilar o modelo
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
model_lstm.fit(X_train, y_train, batch_size=1, epochs=1)

# Fazer previsões com o modelo
train_predict = model_lstm.predict(X_train)
test_predict = model_lstm.predict(X_test)

# Reverter a normalização
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Garantir que as previsões estejam dentro do limite de dados disponíveis
def get_prediction(test_predict, step):
    if step < len(test_predict):
        return test_predict[step]
    else:
        return test_predict[-1]  # Retorna o último valor disponível se o índice for maior

# Previsão para intervalos específicos com ajuste de tamanho
prevision_5_min = get_prediction(test_predict, 0)
prevision_15_min = get_prediction(test_predict, 2)
prevision_30_min = get_prediction(test_predict, 5)
prevision_60_min = get_prediction(test_predict, 11)
prevision_1440_min = get_prediction(test_predict, 287)  # 1 dia
prevision_43200_min = get_prediction(test_predict, 8639)  # 30 dias

# Criar JSON com as previsões
predictions_lstm = {
    "prevision_5_min": float(prevision_5_min[0]),
    "prevision_15_min": float(prevision_15_min[0]),
    "prevision_30_min": float(prevision_30_min[0]),
    "prevision_60_min": float(prevision_60_min[0]),
    "prevision_1440_min": float(prevision_1440_min[0]),
    "prevision_43200_min": float(prevision_43200_min[0]),
    "stop_loss": float(test_predict.min() * 0.98),
    "pull_back": float(test_predict.max() * 1.02),
    "risk_management_factor": 1.5,
    "recommended_investment_amount": 1000
}

# Salvar previsões em JSON
with open('predictions_with_lstm.json', 'w') as json_file:
    json.dump(predictions_lstm, json_file)

print("Previsões salvas em 'predictions_with_lstm.json'")
