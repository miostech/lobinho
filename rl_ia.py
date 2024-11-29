import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import json


class TradingEnv(gym.Env):
    """
    Custom Environment for Reinforcement Learning in Trading with Leverage
    """

    def __init__(self, data, leverage=10, margin_call=0.25, max_loss_per_action=0.05):
        super(TradingEnv, self).__init__()

        # Dados de mercado
        self.data = data
        self.current_step = 0

        # Parâmetros financeiros
        self.balance = 100000
        self.positions = 0
        self.leverage = leverage  # Alavancagem
        self.margin_call = margin_call  # Margem mínima (25% do saldo inicial)
        self.max_loss_per_action = max_loss_per_action  # Perda máxima por ação (5% do saldo inicial)
        self.initial_balance = self.balance

        self.max_steps = len(self.data) - 1

        # Ações possíveis: [0 - Manter, 1 - Comprar, 2 - Vender]
        self.action_space = spaces.Discrete(3)

        # Observação: Preços (open, high, low, close, volume) e saldo
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),  # 5 preços do mercado + saldo
            dtype=np.float32
        )

    def reset(self):
        """
        Reseta o ambiente para o início do episódio.
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = 0
        return self._get_observation()

    def _get_observation(self):
        """
        Retorna o estado atual.
        """
        market_data = self.data.iloc[self.current_step][['open', 'high', 'low', 'close', 'baseVolume']].values
        return np.append(market_data, self.balance).astype(np.float32)

    def step(self, action):
        """
        Realiza uma ação e retorna o próximo estado, recompensa, e se o episódio terminou.
        """
        current_price = self.data.iloc[self.current_step]['close']

        # Ações
        if action == 1:  # Comprar
            # Verifica margem disponível
            required_margin = current_price * self.leverage
            if self.balance >= required_margin:
                self.positions += self.leverage
                self.balance -= required_margin
        elif action == 2:  # Vender
            if self.positions >= self.leverage:
                self.positions -= self.leverage
                self.balance += current_price * self.leverage

        # Atualiza o passo
        self.current_step += 1

        # Calcula equity
        equity = self.balance + (self.positions * self.data.iloc[self.current_step]['close'])

        # Recompensa baseada na mudança no equity
        reward = equity - (self.balance + (self.positions * current_price))

        # Penaliza caso a perda exceda o máximo permitido por ação
        max_allowed_loss = self.max_loss_per_action * self.initial_balance
        if reward < -max_allowed_loss:
            reward -= 1000  # Penalidade adicional

        # Penalidade por chamada de margem
        if equity < self.margin_call * self.initial_balance:
            # Liquida posições
            self.positions = 0
            self.balance = equity
            done = True
            reward -= 1000  # Penalidade severa
        else:
            done = self.current_step >= self.max_steps

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        """
        Renderiza o ambiente (para debug).
        """
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance:.2f}, Positions: {self.positions}")


def preprocess_json_data(json_data):
    """
    Preprocessa os dados do JSON para o ambiente.
    """
    df = pd.DataFrame(json_data)
    df = df[['open', 'high', 'low', 'close', 'baseVolume']]
    return df


# Carregar os dados do JSON
with open('ccxt.json') as f:
    json_data = json.load(f)

# Dados de treino
data = preprocess_json_data(json_data)

# Criar o ambiente com alavancagem
leverage = 10  # Alavancagem de 10x
env = TradingEnv(data, leverage=leverage)

# Criar o modelo PPO
model = PPO("MlpPolicy", env, verbose=1)

# Treinar o modelo
model.learn(total_timesteps=10000)

# Avaliar o modelo
obs = env.reset()
for _ in range(len(data)):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        break
