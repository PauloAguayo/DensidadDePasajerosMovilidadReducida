# DensidadDePasajerosMovilidadReducida
El presente trabajo estudia el efecto de la densidad de pasajeros en el espacio ocupado por personas con movilidad reducida, específicamente personas en sillas de ruedas.

En esta entrega, se utilizó el poder de las redes neuronales, más específicamente de Deep Learning, en el que se usó la red Faster RCNN con el modelo InceptionV2. La red fue entrenada con imágenes de los experimentos ejecutados en el Laboratorio de Dinámica Humana de la Universidad de Los Andes (Chile), MobilityAids Dataset (Autonome Intelligente Systeme, Albert-Ludwigs-Universität Freiburg - http://mobility-aids.informatik.uni-freiburg.de/), paper "Head and Shoulder Detection using Convolutional Networks and RGBD data" (http://www.site.uottawa.ca/research/viva/projects/head_shoulder_detection/index.html), y del canal de Youtube UWSpinalCordInjury.

La red fue entrenada a través de Google Cloud Platform con una tarjeta GPU Tesla T4, con una duración de aproximadamente 7-8 horas. Esta red está encargada de detectar cabezas y personas en sillas de ruedas.

# Instalación 
Se debe generar un ambiente virtual con las siguientes librerías:

- Python >=3 
- Tensorflow 1.14
- OpenCV
- Shapely
- Matplotlib

# Parser
El programa posee 9 variables parser, de las cuales sólo 4 son obligatorias. Se describen a continuación:

- '-m' (obligatoria): Ruta y nombre del modelo ".pb".
- '-l' (obligatoria): Ruta y nombre de las etiquetas o labels ".pbtxt".
- '-i' (obligatoria): Ruta, nombre y formato de la imagen.
- '-mode' (obligatoria): "1" para elegir como ground truth el área de sillas de ruedas, "2" el área del polígono como ground truth. 
- '-o' (opcional): Ruta, nombre y formato de la imagen de salida.
- '-wr' (opcional): Valor en metros del ancho de una silla de ruedas. Su valor por default es 0.8 metros.
- '-ws' (opcional): Valor en metros del largo lateral de una silla de ruedas. Su valor por default es 1.2 metros.
- '-c' (opcional): Probabilidad de detección de objetos. Default 0.8.
- '-d' (opcional): Discriminante de distancias al centroide del polígono. Default 0.7.

# Usos
- '-mode' 1: Esta selección va relacionada con las variables parser '-wr', y '-ws', opcionalmente.
- '-mode' 2: En esta elección se preguntará por el área del polígono, a modo de obtener su valor ground truth. En este modo, las variables '-wr' y '-ws' no son útiles, es decir, no son consideradas en ningún cálculo.

# Ejemplo
$ python Object_detection_image.py -i frames1_0000000069.png -m inference_graph/frozen_inference_graph.pb -l training/labelmap.pbtxt -mode 1

# Agradecimientos 
- https://docs.opencv.org/master/db/d5b/tutorial_py_mouse_handling.html
- https://github.com/tensorflow/models
- https://github.com/ZidanMusk/experimenting-with-sort
