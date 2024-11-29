import gymnasium as gym  # Atualização para gymnasium
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

# Função para inicializar o ambiente
def initialize_environment(data, window_size=60, min_profit_threshold=0.1, return_period=3):
    environment = {
        'data': data,
        'current_step': window_size,
        'window_size': window_size,
        'min_profit_threshold': min_profit_threshold,  # Percentual de lucro esperado, ex.: 10%
        'return_period': return_period  # Período (em dias) para alcançar o lucro
    }
    return environment

# Função para observar o próximo estado (retorna um numpy array)
def next_observation(environment):
    current_step = environment['current_step']
    window_size = environment['window_size']
    data = environment['data']

    window = data.iloc[current_step - window_size:current_step]

    # Inclui dados financeiros e sentimento e retorna como array concatenado
    obs = np.concatenate([
        window['open'].values,
        window['high'].values,
        window['low'].values,
        window['close'].values,
        window['volume'].values,
        window['relevance_score'].values,
        window['ticker_sentiment_score'].values
    ]).astype(np.float32)  # Converter para float32

    return obs

# Função para projetar o preço futuro
def project_future_price(environment):
    future_step = environment['current_step'] + environment['return_period']
    data = environment['data']

    if future_step >= len(data):
        future_price = data.iloc[environment['current_step']]['close']
    else:
        future_price = data.iloc[future_step]['close']

    return future_price

# Função para calcular o lucro esperado
def expected_profit(current_price, future_price):
    return (future_price - current_price) / current_price

# Função para executar uma ação (comprar, vender ou manter)
def execute_action(environment, action):
    current_price = environment['data'].iloc[environment['current_step']]['close']
    future_price = project_future_price(environment)

    profit = 0.0  # Definir um valor padrão para o profit

    if action == 1:  # Comprar
        profit = expected_profit(current_price, future_price)

        if profit >= environment['min_profit_threshold']:
            print(f"Lucro esperado de {profit:.2f} em {environment['return_period']} dias. Compra aprovada.")
        else:
            print(f"Lucro esperado de {profit:.2f} é menor que o mínimo de {environment['min_profit_threshold']:.2f}. Não comprando.")

    elif action == 2:  # Vender
        print(f"Vendendo ação ao preço atual de {current_price}.")

    # Avançar para o próximo passo
    environment['current_step'] += 1

    # Verificar se o episódio terminou
    terminated = environment['current_step'] >= len(environment['data']) - 1

    # Truncated indica se o episódio foi interrompido por outros fatores (usaremos False por padrão)
    truncated = False

    # Info é um dicionário com informações extras (neste caso, está vazio)
    info = {}

    return next_observation(environment), profit, terminated, truncated, info

# Ambiente usando Gymnasium
class TradingEnv(gym.Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()

        self.environment = initialize_environment(data)
        self.action_space = spaces.Discrete(3)  # 0 = manter, 1 = comprar, 2 = vender

        # Ajustar o espaço de observação para permitir valores negativos
        num_columns = 7  # 7 colunas: open, high, low, close, volume, relevance_score, ticker_sentiment_score
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_columns * self.environment['window_size'],), dtype=np.float32)

    def step(self, action):
        obs, profit, terminated, truncated, info = execute_action(self.environment, action)
        reward = profit if action == 1 and profit >= self.environment['min_profit_threshold'] else 0
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.environment = initialize_environment(self.environment['data'])
        return next_observation(self.environment), {}

    def render(self):
        # Exibir o estado atual do ambiente
        current_step = self.environment['current_step']
        current_price = self.environment['data'].iloc[current_step]['close']
        print(f"Step: {current_step}, Preço Atual: {current_price}")

# Função principal para executar o código e treinar o agente
def main():

    # Carregar os dados CSV
    data = pd.read_csv('merge_dataframes.csv')

    env = TradingEnv(data)

    # Verificar se o ambiente está correto
    check_env(env)

    # Criar o modelo DQN
    model = DQN('MlpPolicy', env, verbose=1)

    # Treinar o modelo
    model.learn(total_timesteps=10000)

    # Testar o modelo treinado
    obs, _ = env.reset()  # Capture apenas a observação, ignore o `info`
    for _ in range(len(data) - env.environment['window_size']):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()  # Chamar a função render corretamente
        if terminated:
            break

# Executar a função main se o arquivo for executado diretamente
if __name__ == "__main__":
    main()
