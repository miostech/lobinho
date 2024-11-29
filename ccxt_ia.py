import ccxt
import time
import json
import os
import datetime

# Cria uma instância da exchange Bybit
exchange = ccxt.bybit()

# Define o par de moedas
symbol = 'BTC/USDT'

# Nome do arquivo JSON
filename = 'ccxt.json'

# Cria o arquivo JSON vazio como uma lista, se ainda não existir
if not os.path.exists(filename):
    with open(filename, 'w') as f:
        json.dump([], f)

# Loop para obter dados em tempo real
while True:
    try:
        # Obtém o ticker para o par especificado
        ticker = exchange.fetch_ticker(symbol)
        # add timestamp and datetime
        ticker['timestamp'] = time.time()
        ticker['datetime'] = datetime.datetime.now().isoformat()

        # Carrega os dados existentes
        with open(filename, 'r') as f:
            data = json.load(f)

        # Adiciona o novo ticker aos dados
        data.append(ticker)

        # Salva os dados atualizados no arquivo
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

        # Exibe o preço atual (opcional)
        last_price = ticker['last']
        print(f'Preço atual do {symbol}: {last_price}')

        # Aguarda 5 segundos antes de buscar novamente
        time.sleep(1)

    except Exception as e:
        print(f'Ocorreu um erro: {e}')
        break
