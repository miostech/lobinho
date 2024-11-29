import datetime
import json
import pandas as pd
import download_data_tiingo
import treat_data_downloaded
import treat_intraday_data
import talib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def treat_date_from_merge(merge_frame: pd.DataFrame) -> pd.DataFrame:
    df = merge_frame
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['dayofyear'] = df['datetime'].dt.dayofyear
    return df

def treat_data_news_sentimental() -> pd.DataFrame:
    print("treat_data_news_sentiment")
    path_to_file = "sample_data_news_sentiment.json"

    with open(path_to_file) as json_file:
        json_data = json.load(json_file)

    json_data = json_data['feed']

    json_data_treated = []
    for data in json_data:
        # Filtra apenas as notícias que contêm o ticker desejado
        ticker_sentiment_list = list(filter(lambda x: x['ticker'] == 'NVDA', data['ticker_sentiment']))
        if ticker_sentiment_list:
            ticker_sentiment = ticker_sentiment_list[0]
            json_data_treated.append({
                'datetime': data['time_published'],
                'ticker': ticker_sentiment['ticker'],
                'relevance_score': ticker_sentiment['relevance_score'],
                'ticker_sentiment_score': ticker_sentiment['ticker_sentiment_score'],
                'ticker_sentiment_label': ticker_sentiment['ticker_sentiment_label'],
            })

    df = pd.DataFrame(json_data_treated)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.astype({'ticker_sentiment_score': 'float'})

    return df

def calculate_rsi_talib(data, rsi_period=6):
    data['RSI'] = talib.RSI(data['close'], timeperiod=rsi_period)
    return data

def calculate_macd_talib(data, short_period=6, long_period=13, signal_period=4):
    macd, signal, hist = talib.MACD(data['close'],
                                    fastperiod=short_period,
                                    slowperiod=long_period,
                                    signalperiod=signal_period)
    data['MACD'] = macd
    data['Signal_Line'] = signal
    data['MACD_Histogram'] = hist
    return data

def calculate_bollinger_bands(data, period=10, std_dev=2):
    upper, middle, lower = talib.BBANDS(data['close'], timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev, matype=0)
    data['Bollinger_Upper'] = upper
    data['Bollinger_Middle'] = middle
    data['Bollinger_Lower'] = lower
    return data


def main(
        value_to_invest: float = 600,
        value_to_stop_loss: float = 10,
        value_to_take_profit: float = 15,
        leverage: int = 20,
        symbol_to_invest: str = 'NVDA',
        year_month: str = "2024-08-01"
):
    """
    Main function to handle AI
    """
    print("init handle_ai")
    print("download_data_tiingo")
    print(f"Downloading data for {symbol_to_invest} in {year_month}.")
    # download_data_tiingo.download_data_tiingo(symbol_to_invest, year_month)
    print("end download")
    print("end handle_ai")
    data_tiingo_treated = treat_data_downloaded.treat_data_downloaded(symbol_to_invest)

    data_frame_intraday_merged = pd.DataFrame()

    for data_frame in data_tiingo_treated:
        data_frame_intraday = treat_intraday_data.treat_intraday_data(data_frame)
        data_frame_intraday_merged = pd.concat([data_frame_intraday_merged, data_frame_intraday])

    data_frame_news_sentimental = treat_data_news_sentimental()

    data_frame_intraday = data_frame_intraday_merged.sort_values(by='datetime')
    data_frame_news_sentimental = data_frame_news_sentimental.sort_values(by='datetime')

    data_frame_intraday['datetime'] = pd.to_datetime(data_frame_intraday['datetime']).dt.tz_convert('UTC')
    data_frame_news_sentimental['datetime'] = pd.to_datetime(data_frame_news_sentimental['datetime']).dt.tz_localize('UTC')

    merge_dataframes = pd.merge_asof(
        data_frame_intraday.sort_values('datetime'),
        data_frame_news_sentimental.sort_values('datetime'),
        on='datetime',
        direction='nearest'
    )

    merge_dataframes = treat_date_from_merge(merge_dataframes)

    # Ordena os dados em ordem ascendente de data
    merge_dataframes = merge_dataframes.sort_values(by='datetime', ascending=True)

    # Salva o DataFrame mesclado
    merge_dataframes.to_csv('merge_dataframes.csv', index=False)

    data = pd.read_csv('merge_dataframes.csv')
    data['datetime'] = pd.to_datetime(data['datetime'])

    data.drop(columns=['date', 'month', 'day', 'dayofyear'], inplace=True)

    # Certifique-se de que os dados estão ordenados em ordem ascendente antes de calcular os indicadores
    data = data.sort_values(by='datetime', ascending=True)

    # Calcula os indicadores técnicos com períodos reduzidos
    data = calculate_rsi_talib(data, rsi_period=6)
    data = calculate_macd_talib(data, short_period=6, long_period=13, signal_period=4)
    data = calculate_bollinger_bands(data, period=10, std_dev=2)

    # Remove linhas com valores NaN resultantes dos cálculos iniciais
    data.dropna(inplace=True)

    # Exibe as primeiras linhas do DataFrame
    print(data.head())

    # print as json data
    print(json.dumps({
        "value_to_invest": value_to_invest,
        "value_to_stop_loss": value_to_stop_loss,
        "value_to_take_profit": value_to_take_profit,
        "leverage": leverage,
        "list_values": data.tail(200).to_json(orient='records', date_format='iso')
    }, indent=4))

    data['MA20'] = talib.SMA(data['close'], timeperiod=20)
    data['RSI'] = talib.RSI(data['close'], timeperiod=14)
    data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = talib.MACD(data['close'])

    # Gráfico de preços com MA
    plt.figure(figsize=(14, 7))
    plt.plot(data['datetime'], data['close'], label='Preço de Fechamento')
    plt.plot(data['datetime'], data['MA20'], label='MA20', color='red')

    # Ajusta o formato das datas no eixo x para exibir data e hora
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

    # Rotaciona os rótulos do eixo x para melhorar a legibilidade
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # Ajusta automaticamente o espaçamento
    plt.xticks(rotation=45)

    # Títulos e legendas
    plt.title('Preço de Fechamento com MA20')
    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Ajusta layout para evitar sobreposição de textos
    plt.show()

    # Criar pairplot para explorar relações entre indicadores e preço
    sns.pairplot(data, vars=['close', 'RSI', 'MACD', 'MACD_Hist'], kind='scatter')
    plt.show()

    # Filtrar dados em condições específicas (exemplo de compra)
    compra = data[(data['RSI'] < 50) & (data['RSI'] > 30) & (data['MACD'] > 0)]

    # Criar gráficos para explorar
    plt.figure(figsize=(12, 6))
    plt.scatter(compra['RSI'], compra['MACD_Hist'], c=compra['close'], cmap='viridis')
    plt.colorbar(label='Preço de Fechamento')
    plt.title('Condições Ideais de Compra')
    plt.xlabel('RSI')
    plt.ylabel('MACD_Hist')
    plt.show()

    # Criar um heatmap das correlações
    corr = data[['close', 'RSI', 'MACD', 'MACD_Hist']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlação entre Indicadores e Preço')
    plt.show()

    # Cenário de compra baseado em condições
    data['compra'] = (data['RSI'] < 50) & (data['MACD'] > data['MACD_Hist'])

    # Filtrar momentos de compra
    momentos_de_compra = data[data['compra']]

    # Atualizando o gráfico para incluir as condições de compra
    plt.figure(figsize=(14, 7))

    # Plotando o preço de fechamento
    plt.plot(data['datetime'], data['close'], label='Preço de Fechamento', color='blue', linewidth=1)

    # Adicionando os momentos de compra baseados em RSI, MACD e MACD_Hist
    plt.scatter(momentos_de_compra['datetime'], momentos_de_compra['close'],
                color='green', label='Sinal de Compra', s=50)

    # Adicionando um título mais descritivo
    plt.title('Momentos de Compra com Base em RSI, MACD e MACD_Hist', fontsize=16)

    # Rotulando os eixos
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Preço', fontsize=12)

    # Legenda
    plt.legend()

    # Exibindo o gráfico
    plt.grid(True)  # Adicionar uma grade para melhor leitura
    plt.show()


__init__ = main
if __name__ == '__main__':
    _value_to_invest = 600
    _value_to_stop_loss = 10
    _value_to_take_profit = 15
    _leverage = 20
    _symbol_to_invest = 'NVDA'
    # Ano atual
    _current_year = datetime.datetime.now().year
    # Seis meses atrás
    _six_months_ago = datetime.datetime.now() - datetime.timedelta(days=30)
    _year_month = f"{_current_year}-{_six_months_ago.month}-01"
    main(_value_to_invest, _value_to_stop_loss, _value_to_take_profit, _leverage, _symbol_to_invest, _year_month)
