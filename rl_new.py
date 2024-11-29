import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from tensorflow.keras.callbacks import EarlyStopping

# Carregar o dataset CSV
data = pd.read_csv('merge_dataframes.csv')

# Remove primeiras 1000 linhas
data = data.iloc[1000:]

# Converter a coluna 'datetime' para datetime
data['datetime'] = pd.to_datetime(data['datetime'])

# Filtrar as colunas que são importantes para a previsão
data = data[['datetime', 'close', 'ticker_sentiment_score']]

# Ordenar os dados pela data
data = data.sort_values('datetime')

# Definir um índice de tempo para facilitar a plotagem
data.set_index('datetime', inplace=True)

# Escalar as colunas 'close' e 'ticker_sentiment_score' separadamente
close_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = close_scaler.fit_transform(data[['close']])

sentiment_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_sentiment = sentiment_scaler.fit_transform(data[['ticker_sentiment_score']])

# Combinar as colunas escalonadas
scaled_data = np.hstack((scaled_close, scaled_sentiment))

# Dividir os dados em treinamento e teste (80% treino, 20% teste)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Função para criar a sequência de dados para o LSTM
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])  # Captura todas as features
        y.append(data[i + time_step, 0])  # Prever a coluna 'close', que está na posição 0
    return np.array(X), np.array(y)

# Criar sequências para treino e teste
time_step = 1000
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Remodelar os dados para [samples, time_steps, features]
num_features = scaled_data.shape[1]  # Temos 2 features: 'close' e 'ticker_sentiment_score'
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], num_features)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], num_features)

# Criar o modelo LSTM
model = Sequential()
model.add(LSTM(200, return_sequences=True, input_shape=(time_step, num_features)))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))  # Prever apenas 'close'

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Treinar o modelo
model.fit(X_train, y_train, batch_size=64, epochs=100, validation_split=0.2, verbose=1, callbacks=[early_stop])

# Fazer previsões para os dados de treino e teste
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Reverter a normalização da previsão 'close' para ver os valores reais
train_predict = close_scaler.inverse_transform(train_predict)
test_predict = close_scaler.inverse_transform(test_predict)

# Reverter a normalização dos valores reais de y_test para comparação
y_test = close_scaler.inverse_transform(y_test.reshape(-1, 1))

# Plotar os resultados
train_data_plot = np.empty_like(scaled_close)
train_data_plot[:, :] = np.nan
train_data_plot[time_step:len(train_predict) + time_step, 0] = train_predict.flatten()

test_data_plot = np.empty_like(scaled_close)
test_data_plot[:, :] = np.nan
test_data_plot[len(train_predict) + (time_step * 2) + 1:len(scaled_close) - 1, 0] = test_predict.flatten()

# Obter os valores máximos e mínimos de 'close'
close_max = data['close'].max()
close_min = data['close'].min()
print(f"Close Max: {close_max}")
print(f"Close Min: {close_min}")

# Calcular RMSE e MAE
rmse = sqrt(mean_squared_error(y_test, test_predict))
mae = mean_absolute_error(y_test, test_predict)
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")

plt.plot(close_scaler.inverse_transform(scaled_close), label='Preço Real')  # Apenas a coluna 'close'
plt.plot(train_data_plot, label='Previsão de Treino')
plt.plot(test_data_plot, label='Previsão de Teste')
plt.legend()
plt.show()

# ------------------------------------------
# Prever o valor de fechamento daqui a 1 hora
# ------------------------------------------

# Últimos 'time_step' valores para prever o próximo ponto no tempo
last_60_days = test_data[-time_step:]

# Redimensionar para o formato [samples, time_steps, features]
last_60_days_scaled = last_60_days.reshape(1, time_step, num_features)

# Fazer a previsão usando o modelo
predicted_price = model.predict(last_60_days_scaled)

# Reverter a escala da previsão para o valor original
predicted_price = close_scaler.inverse_transform(predicted_price)

print(f"Preço previsto daqui a 1 hora: {predicted_price[0][0]}")

# ------------------------------------------
# Previsão para vários períodos futuros (ex: próximo 1 hora)
# ------------------------------------------

future_predictions = []
current_input = last_60_days_scaled.copy()  # Faça uma cópia para evitar alterar o original

# Prever para as próximas 30 janelas de tempo (30 intervalos)
for i in range(30):
    predicted_price = model.predict(current_input)

    # Reverter a normalização para a previsão (apenas para a coluna 'close')
    predicted_price_unscaled = close_scaler.inverse_transform(predicted_price)

    # Adicionar a previsão à lista
    future_predictions.append(predicted_price_unscaled[0][0])

    # Manter o valor de 'ticker_sentiment_score' da última entrada
    last_sentiment_score = current_input[0, -1, 1]  # Último valor de sentiment score

    # Criar o array com as previsões e o valor 'ticker_sentiment_score'
    predicted_price_combined = np.array([[predicted_price_unscaled[0][0], last_sentiment_score]])

    # Reescalar a previsão para reutilizar como entrada
    predicted_price_scaled = np.hstack((close_scaler.transform(predicted_price_unscaled.reshape(-1, 1)),
                                        sentiment_scaler.transform(np.array([[last_sentiment_score]]))))

    # Atualizar a entrada para incluir a nova previsão (descartar o valor mais antigo)
    current_input = np.append(current_input[:, 1:, :], predicted_price_scaled[:, np.newaxis, :], axis=1)


# Mostrar as previsões para os próximos 30 períodos
print("Previsões para o próximo período:")
for i, prediction in enumerate(future_predictions):
    print(
        f"Período {i + 1}: {prediction} data atual: {data.index[-1]} data futura: {data.index[-1] + pd.Timedelta(minutes=5 * (i + 1))}")
