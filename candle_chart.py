import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import talib as ta

# Carregar os dados
df = pd.read_csv('merge_dataframes.csv', parse_dates=True, index_col='datetime')

# Limitar para os 200 últimos dados
df = df[:200]

# Ordenar os dados por datetime de forma ascendente
df = df.sort_values(by='datetime', ascending=True)

# Calcular indicadores técnicos
df['RSI'] = ta.RSI(df['close'], timeperiod=14)
df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['upper_band'], df['middle_band'], df['lower_band'] = ta.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

# Verificar se há valores NaN
print(df[['MACD', 'MACD_signal', 'MACD_hist']].isna().sum())

# Remover linhas com valores NaN em qualquer uma das colunas calculadas para garantir que os gráficos estejam sincronizados
df.dropna(subset=['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'upper_band', 'middle_band', 'lower_band'], inplace=True)

# Configuração do gráfico candlestick
mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
s = mpf.make_mpf_style(marketcolors=mc)

# Criar subplots para candlestick, RSI, MACD e Bandas de Bollinger
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1, 1]})

# Gráfico de Candlestick com Bandas de Bollinger
apds = [
    mpf.make_addplot(df['upper_band'], ax=ax1, color='blue'),
    mpf.make_addplot(df['middle_band'], ax=ax1, color='orange'),
    mpf.make_addplot(df['lower_band'], ax=ax1, color='blue')
]

mpf.plot(df, type='candle', ax=ax1, style=s, addplot=apds)
ax1.set_ylabel('Price')

# Gráfico de RSI
ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
ax2.axhline(70, color='red', linestyle='--')
ax2.axhline(30, color='green', linestyle='--')
ax2.set_ylabel('RSI')

# Gráfico de MACD com ajuste de limites no eixo Y e maior largura de histograma
ax3.plot(df.index, df['MACD'], label='MACD', color='blue')
ax3.plot(df.index, df['MACD_signal'], label='Signal Line', color='red')
ax3.bar(df.index, df['MACD_hist'], label='MACD Hist', color='gray', width=0.02)
ax3.set_ylim(df['MACD_hist'].min() - 0.1, df['MACD_hist'].max() + 0.1)  # Ajustar o eixo Y para melhorar a visualização
ax3.set_ylabel('MACD')
ax3.legend(loc='upper left')  # Legenda para o MACD

# Adiciona título ao gráfico
ax1.set_title('Candlestick Chart with RSI and MACD')

# Ajustes finais
plt.tight_layout()
plt.show()
