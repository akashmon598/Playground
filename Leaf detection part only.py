
#IMPORTING GUI MODULES #
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication,QWidget, QVBoxLayout, QPushButton, QFileDialog , QLabel, QTextEdit,QMainWindow
import sys 
from PyQt5.QtGui import QPixmap

#IMPORTING IMAGE PROCESSING MODULES #
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
#IMPORTING CAMERA MODULES #
#import time 
#from pysony import SonyAPI, ControlPoint, common_header

# THIS IS THE SECOND WINDOW #
class AnotherWindow(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        # THIS BUTTON IS USED TO CONTROL THE ZOOM OF THE CAMERA #
        
        self.zoom_button = QPushButton(" ZOOM IN ")
        layout.addWidget(self.zoom_button)
        
        # THIS BUTTON IS USED TO CAPTURE THE IMAGE FROM THE CAMERA #
        
        self.capture_button = QPushButton(" CAPTURE ")
        layout.addWidget(self.capture_button)

# THIS IS THE FIRST WINDOW #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        global layout
        layout = QVBoxLayout(central_widget)
        
        # BUTTON TO CONTROL THE CAMERA #
        self.new_window = QPushButton("CAMERA CONTROLS")
        self.new_window.clicked.connect(self.show_new_window)
        layout.addWidget(self.new_window)
        
        # BUTTON TO OPEN THE IMAGE #
        self.open_image = QPushButton("SELECT IMAGE ")
        self.open_image.clicked.connect(self.getImage)
        layout.addWidget(self.open_image)
        
        # BUTTON TO PROCESS THE IMAGE AND GET THE OUTPUT #
        self.process_image_button = QPushButton("process")
        self.process_image_button.clicked.connect(self.process_image)
        layout.addWidget(self.process_image_button)
        
        # TO DISPLAY THE IMAGE #
        self.display_image = QLabel("")
        layout.addWidget(self.display_image)
        
        # TO DISPLAY TH NAME OF THE PLANT #
        self.name_of_plant = QLabel("")
        layout.addWidget(self.name_of_plant)
        
        # TO DISPLAY THE NAME OF THE DISEASE #
        self.plant_disease = QLabel("")
        layout.addWidget(self.plant_disease)
        
    def getImage(self):
        fname = QFileDialog.getOpenFileName(self, "Open file", " YOUR FILE DIRECTORY", "Image files (*.jpeg , *.gif) ")
        global imagePath
        imagePath = fname[0]
        pixmap = QPixmap(imagePath)
        print(imagePath)
        self.display_image.setPixmap(QPixmap(pixmap))
        self.resize(pixmap.width(), pixmap.height())

    def show_new_window(self, checked):
        self.w = AnotherWindow()
        self.w.show()

    def process_image(self):
        config = ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        new_model = load_model('YOUR FILE DIRECTORY /tomatodisease.h5')
        new_model.summary()
        filename = 'YOUR FILE DIRECTORY/plant_disease_label_transform.pkl'
        image_labels = pickle.load(open(filename, 'rb'))
        DEFAULT_IMAGE_SIZE = tuple((256, 256))
        image = cv2.imread(imagePath)
        image_array  = cv2.resize(image, DEFAULT_IMAGE_SIZE)
        np_image = np.array(image_array, dtype=np.float16) / 225.0
        np_image = np.array(image_array, dtype=np.float16) / 225.0
        np_image = np.expand_dims(np_image,0)
        result = new_model.predict_classes(np_image)
        
        global imager
        global splitter
        global plant_label
        global plant_disease
        
        imager = (image_labels.classes_[result][0])
        splitter = imager.split("___")
        plant_label =   splitter[0]
        plant_disease = splitter[1]
        
        self.name_of_plant = QLabel("PLANT: "+ " " + plant_label)
        layout.addWidget(self.name_of_plant)
        
        self.plant_disease = QLabel("DISEASE: "+ " " + plant_disease)
        layout.addWidget(self.plant_disease)
        
        
app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec_()
