import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def prever_proximo_close_5min(df_atual, model, scaler):
    """
    Usa o último registro do DataFrame para prever o valor de close para os próximos 5 minutos.
    """
    # Obter o último registro do DataFrame
    ultimo_registro = df_atual.iloc[-1].copy()

    # Ajustar o datetime para o próximo período (5 minutos no futuro)
    novo_tempo = pd.to_datetime(ultimo_registro['datetime']) - pd.Timedelta(minutes=60)

    # Atualizar as variáveis temporais
    ultimo_registro['datetime'] = novo_tempo
    ultimo_registro['hour'] = novo_tempo.hour
    ultimo_registro['day'] = novo_tempo.day
    ultimo_registro['dayofweek'] = novo_tempo.dayofweek
    ultimo_registro['dayofyear'] = novo_tempo.dayofyear
    ultimo_registro['month'] = novo_tempo.month

    # Criar um DataFrame com o registro ajustado
    proximo_registro = pd.DataFrame([ultimo_registro])

    # Selecionar as colunas necessárias e transformar os dados
    colunas = ['open', 'high', 'low', 'volume', 'relevance_score',
               'ticker_sentiment_score', 'hour', 'dayofweek', 'month', 'day', 'dayofyear']

    X_novo = proximo_registro[colunas]
    X_novo_scaled = scaler.transform(X_novo)

    # Fazer a previsão
    previsao = model.predict(X_novo_scaled)

    return previsao[0][0], novo_tempo

# Carregar os dados, o modelo e o scaler
df = pd.read_csv('merge_dataframes.csv')
model = load_model('NVDA_0.8128516674041748.h5')
scaler = joblib.load('scaler.pkl')

# Converter a coluna 'datetime' para datetime, caso ainda não esteja
df['datetime'] = pd.to_datetime(df['datetime'])

# Obter os últimos 10 valores reais de 'close'
ultimos_10 = df[['datetime', 'close']].tail(100).copy()

# Fazer a previsão para os próximos 5 minutos
proxima_previsao, tempo_previsao = prever_proximo_close_5min(df, model, scaler)

# Adicionar a previsão ao DataFrame
novo_registro = pd.DataFrame({'datetime': [tempo_previsao], 'close': [proxima_previsao]})
df_com_previsao = pd.concat([ultimos_10, novo_registro])

# Configurar o plot para lidar com datas
plt.figure(figsize=(10, 6))
plt.plot(df_com_previsao['datetime'], df_com_previsao['close'], marker='o', linestyle='-', color='b', label='Fechamento Real')
plt.axvline(x=tempo_previsao, color='r', linestyle='--', label='Previsão')
plt.plot(novo_registro['datetime'], novo_registro['close'], marker='o', color='g', label='Fechamento Previsto')

plt.xlabel('Tempo')
plt.ylabel('Valor de Fechamento (Close)')
plt.title('Últimos 10 Valores de Fechamento e Previsão')
plt.legend()
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.tight_layout()
plt.show()
