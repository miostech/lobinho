from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

# Configurar o ChromeOptions para conectar ao Chrome já aberto
chrome_options = Options()
chrome_options.debugger_address = "127.0.0.1:9222"  # Porta onde o Chrome está ouvindo

# Conectar ao Chrome já aberto
driver = webdriver.Chrome(options=chrome_options)

# Acessar o site (se necessário, o site que já está aberto)
url = 'https://www.etoro.com/markets/nsdq100/chart/'
driver.get(url)

# Aguarda um tempo para garantir que todos os elementos sejam carregados
time.sleep(5)

# Localizar o botão usando o atributo 'automation-id' e clicar nele
resize_button = driver.find_element(By.CSS_SELECTOR, '[automation-id="tv-chart-resize-control"]')
resize_button.click()

# Aguarda para garantir que a ação do botão foi processada
time.sleep(2)

# Localizar a div com o id 'tv-chart-container' e em seguida o iframe dentro dela
tv_chart_div = driver.find_element(By.ID, 'tv-chart-container')
iframe = tv_chart_div.find_element(By.TAG_NAME, 'iframe')

# Trocar o foco para o iframe
driver.switch_to.frame(iframe)

# Localizar o elemento do gráfico dentro do iframe. Verifique o seletor CSS do gráfico.
chart_element = driver.find_element(By.CSS_SELECTOR, '.chart-container')  # Ajuste o seletor se necessário
chart_element.screenshot('tradingview_chart.png')  # Salvar o gráfico como PNG

# Finalizar o driver (se necessário)
driver.quit()

print("Gráfico salvo como tradingview_chart.png")
