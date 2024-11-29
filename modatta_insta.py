import cv2
import numpy as np

# Carregar a imagem original e o template
image = cv2.imread('test_2.jpeg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
template = cv2.imread('test_base_gray.png', 0)

# Obter as dimensões do template
template_height, template_width = template.shape

# Variáveis para armazenar a melhor correspondência
best_match_value = -1
best_scale = None
best_location = None

# Iterar sobre múltiplas escalas do template
for scale in np.linspace(0.5, 1.5, 20):  # Escalas de 50% a 150%
    # Redimensionar o template de acordo com a escala
    resized_template = cv2.resize(template, (int(template_width * scale), int(template_height * scale)))

    # Se o redimensionamento for maior que a imagem original, pular
    if resized_template.shape[0] > gray_image.shape[0] or resized_template.shape[1] > gray_image.shape[1]:
        continue

    # Aplicar o Template Matching
    result = cv2.matchTemplate(gray_image, resized_template, cv2.TM_CCOEFF_NORMED)

    # Encontrar o valor máximo de correspondência
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Se o valor de correspondência for melhor do que o melhor anterior, atualizar
    if max_val > best_match_value:
        best_match_value = max_val
        best_location = max_loc
        best_scale = scale
        best_template = resized_template

# Desenhar o retângulo na melhor correspondência encontrada
if best_location:
    top_left = best_location
    bottom_right = (top_left[0] + best_template.shape[1], top_left[1] + best_template.shape[0])
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

print("Precisão da correspondência:", best_match_value)

if best_match_value > 0.9:
    print("Template encontrado!")
else:
    print("Template não encontrado.")

# Mostrar a imagem com a melhor correspondência
cv2.imshow('Multi-Scale Template Matching', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
