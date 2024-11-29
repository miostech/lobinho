from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import re
import os

# Configurar diretório de cache alternativo
os.environ["TRANSFORMERS_CACHE"] = "/Users/oscarsilva/PycharmProjects/transformers_cache"

# Definir o caminho correto para o executável do Tesseract
pytesseract.pytesseract.tesseract_cmd = 'tesseract'

try:
    # Carregar e pré-processar a imagem
    imagem = Image.open("teste_12.jpeg").convert("L")
    enhancer = ImageEnhance.Contrast(imagem).enhance(2)
    imagem = ImageEnhance.Brightness(imagem).enhance(1.5)
    imagem = imagem.filter(ImageFilter.MedianFilter())
    texto_extraido = pytesseract.image_to_string(imagem)

    # Imprimir o texto extraído
    print("Texto extraído da imagem:")
    print(texto_extraido)

    # Verificar se a frase "a seguir" está presente no texto extraído
    if "a seguir" in texto_extraido.lower():
        print("\nO usuário está seguindo a página!")
    else:
        print("\nO usuário NÃO está seguindo a página.")

except pytesseract.TesseractError as e:
    print("Erro do Tesseract:", e)
except UnicodeDecodeError as e:
    print("Erro de decodificação ao processar a saída do Tesseract:", e)
except Exception as e:
    print("Erro geral:", e)
