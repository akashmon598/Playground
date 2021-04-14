from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication,QWidget, QVBoxLayout, QPushButton, QFileDialog , QLabel, QTextEdit
import sys 
from PyQt5.QtGui import QPixmap
import numpy as np
import pickle
import cv2
import os
import matplotlib.pyplot as plt
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import load_model
import time 
from pysony import SonyAPI, ControlPoint, common_header



class Window(QWidget):
	
    def __init__(self):
        super().__init__()
 
        self.title = "TOMATO LEAF DISEASE DETECTOR"
        self.top = 1000
        self.left = 1000
        self.width = 1000
        self.height = 1000
 
 
        self.InitWindow()
 
 
    def InitWindow(self):
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        global vbox
 
        vbox = QVBoxLayout()
 
        self.btn1 = QPushButton("Open Image")
        self.btn1.clicked.connect(self.getImage)
        
        self.btn2 = QPushButton("process")
        self.btn2.clicked.connect(self.process)

        self.btn3 = QPushButton("Capture")
        self.btn3.clicked.connect(self.camera)
        
 
        vbox.addWidget(self.btn1)
        vbox.addWidget(self.btn2)
        vbox.addWidget(self.btn3)
        self.label = QLabel("")
        vbox.addWidget(self.label)
        self.label1 = QLabel("")
        vbox.addWidget(self.label1)
        self.label2 = QLabel("")
        vbox.addWidget(self.label2)
        #self.label3 = QLabel("")
        #vbox.addWidget(self.label3)
        #self.label4 = QLabel("")
        #vbox.addWidget(self.label4)
        
 

        self.setLayout(vbox)
 
        self.show()
 
    def camera(self):
    	search =ControlPoint()
    	cameras =search.discover(1) 
    	camera =SonyAPI(QX_ADDR=cameras[0])
    	camera.setShootMode(["still"])
    	time.sleep(1)
    	camera.setTouchAFPosition(param=[ 23.2, 45.2 ])
    	camera.actTakePicture("1")
    	time.sleep(2)



    def getImage(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            '/home/ucal/Desktop/', "Image files (*.jpg )")
        global imagePath
        imagePath = fname[0]
        global pixmap
        pixmap = QPixmap(imagePath)
        print(imagePath)

        self.label.setPixmap(QPixmap(pixmap))
        self.resize(pixmap.width(), pixmap.height())
        

        
    def process(self):
        config = ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        new_model = load_model('/home/akash/Desktop/myfiles/GUIcode/sony_camera_api-develop/examples/tomatodisease.h5')
        new_model.summary()
        filename = '/home/akash/Desktop/myfiles/GUIcode/sony_camera_api-develop/examples/plant_disease_label_transform.pkl'
        image_labels = pickle.load(open(filename, 'rb'))
        DEFAULT_IMAGE_SIZE = tuple((256, 256))
        image = cv2.imread(imagePath)
        image_array  = cv2.resize(image, DEFAULT_IMAGE_SIZE)
        np_image = np.array(image_array, dtype=np.float16) / 225.0
        np_image = np.array(image_array, dtype=np.float16) / 225.0
        np_image = np.expand_dims(np_image,0)
        result = new_model.predict_classes(np_image)
        
        
        
        global imager
        global x
        global plant_label
        global plant_disease
        imager = (image_labels.classes_[result][0])
        x = imager.split("___")
        plant_label =x[0]
        plant_disease=x[1]
        #a = (x[0])
        #b = (x[1]) 
        self.label1 = QLabel("PLANT:" + plant_label)
        vbox.addWidget(self.label1)
        self.label2 = QLabel("DISEASE:" + plant_disease)
        vbox.addWidget(self.label2)
        #self.label3 = QLabel()
        #vbox.addWidget(self.label3)
        #self.label4 = QLabel()
        #vbox.addWidget(self.label4)


    
      


    	
    	    







    

 
 
 
App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())
sys.exit(App.exec())
