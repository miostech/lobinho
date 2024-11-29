import json
import pandas as pd
import datetime


def treat_intraday_data(json_data) -> pd.DataFrame:
    print("treat_intraday_data")

    # json_data = json_data['Time Series (5min)']

    json_data_treated = []
    for key in json_data:
        json_data_treated.append({
            # convert to YYYY-MM-DD HH:MM:SS
            'datetime': key['date'],
            'open': key['open'],
            'high': key['high'],
            'low': key['low'],
            'close': key['close'],
            # 'volume': json_data[key]['5. volume']
        })

    df = pd.DataFrame(json_data_treated)
    # remove +0000 from datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float'})

    return df
