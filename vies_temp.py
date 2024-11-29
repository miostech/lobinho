import pandas as pd
from sklearn.model_selection import train_test_split

# Carregar os dados
df = pd.read_csv('merge_dataframes.csv')

# Verificar as primeiras linhas para entender a estrutura
print("Primeiras linhas do DataFrame:")
print(df.head())

# Converter a coluna 'datetime' para o tipo datetime
df['datetime'] = pd.to_datetime(df['datetime'])

# Verificar o intervalo de tempo dos dados
print("\nIntervalo de tempo dos dados:")
print(f"Data mínima: {df['datetime'].min()}")
print(f"Data máxima: {df['datetime'].max()}")

# Ordenar os dados pela coluna 'datetime' para garantir a ordem temporal
df = df.sort_values(by='datetime')

# Verificar se a ordenação foi feita corretamente
print("\nPrimeiras 10 datas após a ordenação:")
print(df['datetime'].head(10))
print("\nÚltimas 10 datas após a ordenação:")
print(df['datetime'].tail(10))

# Dividir os dados de forma aleatória usando train_test_split para verificar se há viés temporal
x_train, x_test, y_train, y_test = train_test_split(df, df['close'], test_size=0.2, random_state=42)

# Examinando as primeiras e últimas datas nos conjuntos de treinamento e teste
print("\nData mínima e máxima no conjunto de treinamento:")
print(f"Treinamento: {x_train['datetime'].min()} - {x_train['datetime'].max()}")
print("\nData mínima e máxima no conjunto de teste:")
print(f"Teste: {x_test['datetime'].min()} - {x_test['datetime'].max()}")

# Verificar a distribuição temporal
print("\nDistribuição temporal no conjunto de treinamento e teste:")
print(f"Datas no treinamento: \n{x_train['datetime'].describe()}")
print(f"Datas no teste: \n{x_test['datetime'].describe()}")
