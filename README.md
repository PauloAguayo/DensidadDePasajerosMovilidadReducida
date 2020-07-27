# DensidadDePasajerosMovilidadReducida
El presente trabajo estudia el efecto de la densidad de pasajeros en el espacio ocupado por personas con movilidad reducida.

En esta entrega, se utilizó el poder de las redes neuronales, más específicamente de Deep Learning, en el que se usó la red Faster RCNN con el modelo InceptionV2. La red fue entrenada con imágenes de los experimentos ejecutados en el Laboratorio de Dinámica Humana de la Universidad de Los Andes (Chile), MobilityAids Dataset (Autonome Intelligente Systeme, Albert-Ludwigs-Universität Freiburg - http://mobility-aids.informatik.uni-freiburg.de/), Head and Shoulder Detection using Convolutional Networks and RGBD data (http://www.site.uottawa.ca/research/viva/projects/head_shoulder_detection/index.html) y del canal de Youtube UWSpinalCordInjury.
La red fue entrenada a través de Google Cloud Platform con una tarjeta GPU Tesla T4, con una duración de aproximadamente 7-8 horas. Esta red está encargada de detectar cabezas y personas en sillas de ruedas.

# Instalación 
Los requisitos para el funcionamiento de del código son los siguientes:

- python >=3 
- Tensorflow 1.14
- OpenCV
- Shapely
- Matplotlib

# Parser
El programa posee 8 variables parser, de las cuales sólo 3 son obligatorias. Se describen a continuación:

- '-m' (obligatoria): Ruta y nombre del modelo ".pb".
- '-l' (obligatoria): Ruta y nombre de las etiquetas o labels ".pbtxt".
- '-i' (obligatoria): Ruta y nombre de la imagen.
- '-o' (opcional): Ruta y/o nombre de la imagen de salida.
- '-wr' (opcional): Valor en metros del ancho de una silla de ruedas. Su valor por default es 0.8 metros.
- '-ws' (opcional): Valor en metros del largo lateral de una silla de ruedas. Su valor por default es 1.2 metros.
- '-c' (opcional): Probabilidad de detección de objetos. Default 0.8.
- '-d' (opcional): Discriminante de distancias al centroide del polígono. Default 0.7.

# Ejemplo
$ python Object_detection_image.py -i frames1_0000000069.png -m inference_graph/frozen_inference_graph.pb -l training/labelmap.pbtxt
