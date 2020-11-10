import cv2

# Este programa reconoce caras en una foto

# Por si queremos pasar el nombre de la imagen como un argumento
#imagePath = sys.argv[1]

# Se carga la imagen y el fichero de cascadas HAAR
# Las cascadas son contenido XML que contiene datos para ayudar a OpenCV a reconocer objetos.
# Al inicializar el código con las cascadas que queremos, OpenCV hará el trabajo por nosotros
# El fichero de cascadas que cargamos reconoce rostros de frente
imagePath = "images/corrs.jpg"
cascPath = "haarcascade_frontalface_default.xml"

# Se crea las cascadas para el reconocimiento de caras
faceCascade = cv2.CascadeClassifier(cascPath)

# Se lee la imagen
image = cv2.imread(imagePath)

# Se convierte a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Se detectan las caras en la imagen (esto devolverá un array con todas las caras detectadas)
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

# Se imprime el número de caras detectadas
print("¡Se han detectado {0} caras!".format(len(faces)))

# Se dibuja un rectángulo alrededor de cada cara
# (x,y) es el punto inicial del rectángulo
# (w,h) es el ancho y alto respectivamente
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Se muestra la imagen con los rectángulos pintados
cv2.imshow("Caras detectadas en la imagen", image)

# La ventana se muestra hasta que el usuario pulsa alguna tecla
cv2.waitKey(0)
