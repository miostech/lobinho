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
    imagem = Image.open("teste_11.jpeg").convert("L")
    enhancer = ImageEnhance.Contrast(imagem).enhance(2)
    imagem = ImageEnhance.Brightness(imagem).enhance(1.5)
    imagem = imagem.filter(ImageFilter.MedianFilter())
    texto_extraido = pytesseract.image_to_string(imagem)

    # Imprimir o texto extraído
    print("Texto extraído da imagem:")
    print(texto_extraido)

    # Lista para armazenar possíveis comentários com nome de usuário e conteúdo
    comentarios = []

    # Ajustar a regex para capturar o nome de usuário, tempo e comentário corretamente
    # A regex agora irá capturar corretamente o comentário, mesmo que ele seja multi-linha
    regex_matches = re.findall(
        r'([a-zA-Z0-9_]+)\s+(?:[1-9][0-9]{0,3}\s*(?:s|sem|min))\s*(.*?)(?=\s*(?:Responder|$))',
        texto_extraido, re.DOTALL
    )

    # Processar os resultados da regex
    for match in regex_matches:
        username = match[0]  # O nome de usuário será o primeiro grupo da regex
        comentario = match[1].strip()  # Captura o texto do comentário e remove espaços extras

        # Ignorar entradas sem um comentário significativo (por exemplo, apenas com "Responder")
        if not comentario or comentario.lower().startswith("responder"):
            continue

        comentarios.append((username, comentario))

    # Exibir os possíveis comentários identificados
    print("\nComentários identificados:")
    for username, comentario in comentarios:
        print(f"Usuário: {username}\nComentário: {comentario}\n")

except pytesseract.TesseractError as e:
    print("Erro do Tesseract:", e)
except UnicodeDecodeError as e:
    print("Erro de decodificação ao processar a saída do Tesseract:", e)
except Exception as e:
    print("Erro geral:", e)


# instagram_account = "teste123"
# instagram_brand = "modatta"
