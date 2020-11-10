import cv2

# Se el fichero de cascadas HAAR
# Las cascadas son contenido XML que contiene datos para ayudar a OpenCV a reconocer objetos.
# Al inicializar el código con las cascadas que queremos, OpenCV hará el trabajo por nosotros
# El fichero de cascadas que cargamos reconoce rostros de frente
cascPath = "haarcascade_frontalface_default.xml"

# Se crea las cascadas para el reconocimiento de caras
faceCascade = cv2.CascadeClassifier(cascPath)

# Se inicia la captura de vídeo
video_capture = cv2.VideoCapture(0)

# El bucle infinito se repite hasta que el usuario salga de la aplicación
# De esta forma, estamos capturando vídeo en tiempo real
while True:
    # Se realiza una captura frame a frame
    ret, frame = video_capture.read()

    # Se convierte la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Se detectan las caras en la imagen (esto devolverá un array con todas las caras detectadas)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Se dibuja un rectángulo alrededor de cada cara
    # (x,y) es el punto inicial del rectángulo
    # (w,h) es el ancho y alto respectivamente
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Se muestra el frame actual
    cv2.imshow('Video', frame)

    # La ventana se cierra cuando el usuario pulsa la tecla q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cuando el usuario pulsa la tecla q y finaliza el bucle
video_capture.release()
cv2.destroyAllWindows()
