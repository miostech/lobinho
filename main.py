import time

import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
import requests
import treat_intraday_data
import treat_data_downloaded


def download_data(symbol: str, year_month: str) -> None:
    print("download_data")
    print(f"Downloading data for {symbol} in {year_month}.")
    # Download data from API
    req = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey=xaLqs5DxYA6zQcfJIbLfD0Rv0lrxGEe7RViAqkuR1gQsnJMhQ7dNiXBPYwZNTALM&month={year_month}")
    data = req.json()
    with open(f"{symbol}_{year_month}.json", "w") as f:
        json.dump(data, f)


def treat_data_news_sentimental() -> pd.DataFrame:
    print("treat_data_news_sentiment")
    path_to_file = "sample_data_news_sentiment.json"

    with open(path_to_file) as json_file:
        json_data = json.load(json_file)

    json_data = json_data['feed']

    json_data_treated = []
    for data in json_data:
        json_data_treated.append({
            'datetime': data['time_published'],
            'ticker': list(filter(lambda x: x['ticker'] == 'NVDA', data['ticker_sentiment']))[0]['ticker'],
            'relevance_score': list(filter(lambda x: x['ticker'] == 'NVDA', data['ticker_sentiment']))[0][
                'relevance_score'],
            'ticker_sentiment_score': list(filter(lambda x: x['ticker'] == 'NVDA', data['ticker_sentiment']))[0][
                'ticker_sentiment_score'],
            'ticker_sentiment_label': list(filter(lambda x: x['ticker'] == 'NVDA', data['ticker_sentiment']))[0][
                'ticker_sentiment_label'],
        })

    df = pd.DataFrame(json_data_treated)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.astype({'ticker_sentiment_score': 'float'})

    return df


def treat_date_from_merge(merge_frame: pd.DataFrame) -> pd.DataFrame:
    df = merge_frame
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['dayofyear'] = df['datetime'].dt.dayofyear
    return df


if __name__ == '__main__':
    # data_frame_intraday = treat_intraday_data()
    symbol = 'BTC'
    data = treat_data_downloaded.treat_data_downloaded(symbol)
    data = data[0][0]['priceData']

    data_frame_intraday_merged = pd.DataFrame()

    for data_frame in data:
        data_frame_intraday = treat_intraday_data.treat_intraday_data(data_frame)
        data_frame_intraday_merged = pd.concat([data_frame_intraday_merged, data_frame_intraday])
    # data_frame_intraday = treat_intraday_data.treat_intraday_data(data[0])
    data_frame_news_sentimental = treat_data_news_sentimental()

    data_frame_intraday = data_frame_intraday_merged.sort_values(by='datetime')
    data_frame_news_sentimental = data_frame_news_sentimental.sort_values(by='datetime')

    merge_dataframes = pd.merge_asof(data_frame_intraday, data_frame_news_sentimental, on='datetime',
                                     direction='nearest')
    merge_dataframes = treat_date_from_merge(merge_dataframes)

    merge_dataframes.to_csv('merge_dataframes.csv', index=False)

    # init training model
    df = pd.read_csv('merge_dataframes.csv')

    features = ['open', 'high', 'low', 'volume', 'relevance_score',
                'ticker_sentiment_score', 'hour', 'dayofweek', 'month', 'day', 'dayofyear']

    # Ordenar os dados pela coluna 'datetime'
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(by='datetime')

    # Dividir os dados com base em uma data de corte para evitar viés temporal
    split_date = '2024-08-25'  # Exemplo de data de corte
    train_data = df[df['datetime'] <= split_date]
    test_data = df[df['datetime'] > split_date]

    x_train = train_data[features]
    y_train = train_data['close']

    x_test = test_data[features]
    y_test = test_data['close']

    point_training = 100

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Defina o tempo máximo de treinamento em segundos (por exemplo, 1 hora = 3600 segundos)
    max_training_time = 3600  # 1 hora
    start_time = time.time()

    # Configurações para detecção de overfitting
    overfitting_limit = 5  # Número de épocas consecutivas com overfitting
    patience = 20  # Número de épocas consecutivas antes de considerar overfitting
    overfitting_count = 0  # Contador para o número de épocas consecutivas com overfitting

    while True:
        model = Sequential([
            Dense(64, activation='relu', kernel_regularizer=l2(0.001), input_shape=(x_train_scaled.shape[1],)),
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
            Dense(1)  # Saída única para a previsão do preço de fechamentoghgg
        ])

        model.compile(optimizer=Adam(learning_rate=0.003), loss='mean_squared_error')

        history = model.fit(x_train_scaled, y_train, epochs=250, validation_data=(x_test_scaled, y_test), batch_size=32,
                            verbose=1)

        test_loss = model.evaluate(x_test_scaled, y_test)
        print(f'Test Loss: {test_loss}')
        model.save(f'{symbol}_{float(test_loss)}.h5')
        # Plotando as curvas de loss de treinamento e validação
        plt.plot(history.history['loss'], label='Treinamento')
        plt.plot(history.history['val_loss'], label='Validação')
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Curvas de Loss de Treinamento e Validação')
        plt.show()

        # Detecção de overfitting
        if history.history['val_loss'][-1] > history.history['loss'][-1] + overfitting_limit:
            overfitting_count += 1
            print(f"Possível overfitting detectado na época {len(history.history['loss'])}.")
            if overfitting_count >= patience:
                print("Overfitting detectado, interrompendo o treinamento.")
                break
        else:
            overfitting_count = 0  # Resetar o contador se não houver overfitting

        time.sleep(0.3)

        # Verifica se o tempo máximo de treinamento foi atingido
        elapsed_time = time.time() - start_time
        if elapsed_time >= max_training_time:
            print("Tempo máximo de treinamento atingido.")
            break
