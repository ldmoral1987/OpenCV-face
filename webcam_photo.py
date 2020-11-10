import time
import cv2
import os

# Este programa toma una foto de la webcam, la graba en un fichero y la muestra por pantalla

# Se accede al recurso de la cámara
camera_port = 0
camera = cv2.VideoCapture(camera_port)

# Este sleep es para darle "algo de tiempo" antes de tomar la captura
# Si se comenta este código, la captura se hará tan rápido que se
# corre el riesgo de que la imagen salga completamente negra
time.sleep(0.001)

# Se lee la imagen
return_value, image = camera.read()

# Se graba la imagen con un nombre (se graba en el directorio del proyecto)
cv2.imwrite("output-frame.png", image)

# Se libera la cámara para que otros procesos puedan utilizarla
del(camera)

# Finalmente, se muestra la imagen grabada
os.startfile("output-frame.png")
