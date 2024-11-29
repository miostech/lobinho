import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt

# Ler o arquivo .csv
data = pd.read_csv('merge_dataframes.csv')

# Ordenar os dados por datetime em ordem decrescente
data = data.sort_values(by='datetime', ascending=False)

# Selecionar as primeiras 100 linhas
data = data.head(100)

# Preparar os dados para o gráfico
data_chart = data[['datetime', 'open', 'high', 'low', 'close']]
data_chart = data_chart.rename(columns={'datetime': 'date'})

# Converter os dados em DataFrame
df = pd.DataFrame(data_chart)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Gerar o gráfico de candlestick com a data completa
mpf.plot(df, type='candle', style='charles', title='Candlestick Chart', ylabel='Price', datetime_format='%Y-%m-%d %H:%M:%S')

# Calcular o RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))

# Plotar o RSI usando Matplotlib
plt.figure(figsize=(10, 5))
plt.plot(rsi, label='RSI', color='blue')
plt.axhline(70, color='red', linestyle='--', label='Sobrecompra (70)')
plt.axhline(30, color='green', linestyle='--', label='Sobrevenda (30)')
plt.title('Índice de Força Relativa (RSI)')
plt.ylabel('RSI')
plt.legend()
plt.show()
