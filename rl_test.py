import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from rl import TradingEnv


# Função para projetar o preço futuro
def project_future_price(environment):
    current_step = environment['current_step']
    return_period = environment['return_period']  # Período em minutos
    data = environment['data']

    future_step = current_step + return_period

    # Verifica se o future_step está além do tamanho do dataset
    if future_step >= len(data):
        future_price = data.iloc[-1]['close']  # Se estiver fora, pega o último preço disponível
        future_datetime = data.iloc[-1]['datetime']  # Última data disponível
    else:
        future_price = data.iloc[future_step]['close']
        future_datetime = data.iloc[future_step]['datetime']  # Data futura

    return future_price, future_datetime


# Função para prever se o preço vai subir ou cair
def prever_subida_ou_queda(model, env, return_period):
    # Ajustar os parâmetros do ambiente para o período de retorno
    env.environment['return_period'] = return_period

    # Reiniciar o ambiente
    obs, _ = env.reset()

    # Fazer a previsão com o modelo treinado
    action, _states = model.predict(obs, deterministic=True)

    # Calcular o preço futuro e obter a data futura
    future_price, future_datetime = project_future_price(env.environment)

    # Preço atual
    current_price = env.environment['data'].iloc[env.environment['current_step']]['close']
    current_datetime = env.environment['data'].iloc[env.environment['current_step']]['datetime']

    # Verificar a ação prevista
    if action == 1:  # Compra
        previsao = "Subir"
    else:
        previsao = "Cair"

    return previsao, current_price, future_price, current_datetime, future_datetime


# Função principal para treinar e testar o modelo
def main():
    # Carregar os dados CSV
    data = pd.read_csv('merge_dataframes.csv')

    # order datetime column in des order
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.sort_values(by='datetime', ascending=False)

    # print the first 5 rows
    print(data.head())

    env = TradingEnv(data)

    # Verificar se o ambiente está correto
    check_env(env)

    # Criar o modelo DQN
    model = DQN('MlpPolicy', env, verbose=1)

    # Treinar o modelo
    model.learn(total_timesteps=10000)

    # Testar o modelo com valores futuros
    tempo_retorno_minutos = 30  # Tempo de retorno em minutos

    previsao, preco_atual, preco_futuro, data_atual, data_futura = prever_subida_ou_queda(model, env, tempo_retorno_minutos)

    print(f"Decisão: O preço vai {previsao} nos próximos {tempo_retorno_minutos} minutos.")
    print(f"Preço atual em {data_atual}: {preco_atual}")
    print(f"Preço previsto em {data_futura}: {preco_futuro}")


# Executar a função main se o arquivo for executado diretamente
if __name__ == "__main__":
    main()
