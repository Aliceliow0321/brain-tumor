
# coding: utf-8

# In[1]:


import sys,os
from PyQt4 import QtCore, QtGui, uic
import PIL
from PIL import Image
from PIL import ImageTk
import cv2,ctypes
import time
from keras import pydot
import numpy as np
import winsound
import matplotlib.pyplot as plt
from keras import backend as K
import pydot_ng as pydot
from PyQt4.QtGui import *
from keras.models import load_model
from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
from keras.preprocessing import image as image_utils

class Example(QtGui.QWidget):
    
    def __init__(self):
        super(Example, self).__init__()
        self.initUI()
        self.pushButton.clicked.connect(self.browseModel)
        self.pushButton_2.clicked.connect(self.LoadInputImage)
        self.pushButton_3.clicked.connect(self.predictOutput)
        self.pushButton_6.clicked.connect(self.layerConfiguration)
        self.pushButton_4.clicked.connect(self.featureMaps)
        self.pushButton_5.clicked.connect(self.individualFeatureMap)
        
        self.model=None
        self.fname = None
        self.layer = None
        self.image=None
        
    def initUI(self):
        
        self.setWindowTitle('Classification of Brain Tumor')
        self.setWindowIcon(QtGui.QIcon('brain.png')) 
        self.pushButton = QtGui.QPushButton('Load Model',self)
        self.pushButton.setGeometry(QtCore.QRect(20, 20, 111, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtGui.QPushButton('Load Input Image',self)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 60, 111, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtGui.QPushButton('Predict Output',self)
        self.pushButton_3.setGeometry(QtCore.QRect(20, 100, 111, 31))
        self.pushButton_3.setObjectName("pushButton_3")
        self.textEdit = QtGui.QTextEdit(self)
        self.textEdit.setGeometry(QtCore.QRect(20, 400, 240, 240))
        self.textEdit.setObjectName("textEdit")
        self.textEdit.setFontPointSize(12)
        self.label_5 = QtGui.QLabel('Input Image : ',self)
        self.label_5.setGeometry(QtCore.QRect(20, 160, 111, 21))
        self.label_5.setObjectName("label_5")
        self.label = QtGui.QLabel(self)
        self.label.setGeometry(QtCore.QRect(20, 180, 200, 200))
        self.label.setObjectName("label")
        self.scrollArea = QtGui.QScrollArea(self)
        self.scrollArea.setGeometry(QtCore.QRect(280, 50, 700, 600))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtGui.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 700, 700))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.label_2 = QtGui.QLabel(self.scrollAreaWidgetContents)
        self.label_2.setGeometry(QtCore.QRect(50, 20, 550, 550))
        self.label_2.setObjectName("label_2")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.label_6 = QtGui.QLabel('Feature Maps : ',self)
        self.label_6.setGeometry(QtCore.QRect(280, 20, 111, 21))
        self.label_6.setObjectName("label_6")
        self.textEdit_2 = QtGui.QTextEdit(self)
        self.textEdit_2.setGeometry(QtCore.QRect(1000, 230, 300, 200))
        self.textEdit_2.setObjectName("textEdit_2") 
        self.textEdit_2.setFontPointSize(12)
        self.label_6 = QtGui.QLabel('Predicted Tumor Type : ',self)
        self.label_6.setGeometry(QtCore.QRect(1040, 460, 111, 21))
        self.label_6.setObjectName("label_6")
        self.textEdit_3 = QtGui.QTextEdit(self)
        self.textEdit_3.setGeometry(QtCore.QRect(1040, 480, 200, 60))
        self.textEdit_3.setObjectName("textEdit_3")
        self.textEdit_3.setFontPointSize(23)
        self.textEdit_3.setTextColor(QColor(255, 0, 0))
        self.spinBox = QtGui.QSpinBox(self)
        self.spinBox.setGeometry(QtCore.QRect(1140, 150, 111, 22))
        self.spinBox.setObjectName("spinBox")
        self.spinBox_2 = QtGui.QSpinBox(self)
        self.spinBox_2.setGeometry(QtCore.QRect(1140, 180, 111, 22))
        self.spinBox_2.setObjectName("spinBox_2")
        self.pushButton_4 = QtGui.QPushButton('Output Feature Maps',self)
        self.pushButton_4.setGeometry(QtCore.QRect(1030, 50, 200, 23))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtGui.QPushButton('Specific Feature Map',self)
        self.pushButton_5.setGeometry(QtCore.QRect(1030, 80, 200, 23))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtGui.QPushButton('Get Layer of Configuration',self)
        self.pushButton_6.setGeometry(QtCore.QRect(1030, 20, 200, 23))
        self.pushButton_6.setObjectName("pushButton_6")
        self.label_3 = QtGui.QLabel('Layer Number : ',self)
        self.label_3.setGeometry(QtCore.QRect(1000, 150, 111, 21))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtGui.QLabel('Feature Maps Number : ',self)
        self.label_4.setGeometry(QtCore.QRect(1000, 180, 121, 21))
        self.label_4.setObjectName("label_4")
         
        self.show()
        
        
    def browseModel(self):
        self.model=None
        self.fname = None
        self.layer = None
        path=os.getcwd()
        model_path = path + '/model'
        self.filePath = QtGui.QFileDialog.getOpenFileName(self,'CNN Model',model_path,'All Files(*.*)')
        self.fname = str(self.filePath)
        print (self.fname)
        print (type(self.fname))
        
        self.textEdit.setText('Model Loaded Successfully'+ '\n' + str(self.fname))
        print ('-------------Loading the Model--------------------')
        self.model=load_model(self.fname)
        print ('-------------Loaded successfully------------------')
        print ('Model Loaded Successfuly')
        self.label.clear() 
        self.label_2.clear()
        self.textEdit_3.clear()
               
        
    def LoadInputImage(self):
        
        self.imageName = None
        self.image=None
        path=os.getcwd()
        test_path = path + '/data/test'
        self.imagePath = QtGui.QFileDialog.getOpenFileName(self,'Load Image',test_path,'All Files(*.*)')
        self.imageName = str(self.imagePath)
        print (self.imageName)
        print (type(self.imageName))
        if len(self.imagePath) > 0:
            imgs=Image.open(self.imagePath)
            self.image= cv2.imread(self.imagePath)
            self.image= cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.resize(self.image,(200,200))
        
        image_profile = QtGui.QImage(self.imagePath)  
        image_profile = image_profile.scaled(200,200, aspectRatioMode=QtCore.Qt.KeepAspectRatio, 
                                             transformMode=QtCore.Qt.SmoothTransformation) 
        self.label.setPixmap(QtGui.QPixmap.fromImage(image_profile)) 
        self.image = cv2.resize(self.image,(128,128))
        self.label_2.clear()
        self.textEdit_3.clear()
       
        
    def predictOutput(self):
        orig = cv2.imread(self.imageName)
        self.image = np.array(self.image)
        self.image = self.image.astype('float32')
        self.image /= 255
        self.image= np.expand_dims(self.image, axis=3)
        self.image= np.expand_dims(self.image, axis=0)
        filepath=os.getcwd()+'/data/test'
        filepath=filepath.replace("\\", "/")
        f1=filepath+'/malignant'
        f2=filepath+'/benign'
        f3=filepath+'/normal'
        head, tail = os.path.split(self.imageName)
        if head == f1:
            real=('Malignant Tumor')
        elif head == f2:
            real=('Benign Tumor')
        elif head ==f3:
            real=('Normal Brain')
        else:
            real=('Unknown')
            error=QtGui.QMessageBox.about(self,'Error',"Non Brain Image")
            self.image.kill()
        print((self.model.predict(self.image)))
        print(self.model.predict_classes(self.image))
        # classify the image
        print("[INFO] classifying image...")
        preds = self.model.predict(self.image)
        preds_class = self.model.predict_classes(self.image)
        m1=np.array_equal(preds_class, [1])
        m2=np.array_equal(preds_class, [0])
         
        if m1 is True:
            label = ('Malignant')
        elif m2 is True:
            label = ('Benign')
        else:
            label = ('Normal')
            
        output_text = ('File : '+ str(self.fname) + '\n' 
                       + 'Class Label :{}'.format(preds_class) + '\n'
                       + 'Class Prob : {}'.format(preds) + '\n'
                       + 'Real Class Name : {}'.format(real) )
        self.textEdit.setText(output_text)
        self.textEdit_3.setText(label)
        
    
    def layerConfiguration(self):
        self.textEdit_2.setText('--------Loading--------')
        self.layer=self.spinBox.text()
        
        def get_featuremaps(model, layer_idx, X_batch):
            get_activations = K.function([model.layers[0].input, K.learning_phase()],
                                         [model.layers[layer_idx].output,])
            activations = get_activations([X_batch,0])
            return activations
        self.activations = get_featuremaps(self.model, int(self.layer),self.image)
        print (np.shape(self.activations))
        feature_maps = self.activations[0][0]
        print (np.shape(feature_maps))
        
        if K.image_dim_ordering()=='th':
            feature_maps=np.rollaxis((np.rollaxis(feature_maps,2,0)),2,0)
                  
        num_of_featuremaps=feature_maps.shape[2]
        fig=plt.figure(figsize=(16,16))
        plt.title("featuremaps-layer-{}".format(self.layer))
        subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
        for i in range(int(num_of_featuremaps)):
            ax = fig.add_subplot(subplot_num, subplot_num, i+1)
            #ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
            ax.imshow(feature_maps[:,:,i],cmap='gray')
            #plt.xticks([])
            #plt.yticks([])
            plt.tight_layout()
            #plt.show()
            fpath=os.getcwd()+'/featuremap/'
            fig.savefig(fpath+"featuremaps-layer-{}".format(self.layer) + '.png')
            
        output_shape = np.shape(self.activations[0])
        featuremap_size = np.shape(self.activations[0][0])[0:2]
        layer_info=self.model.layers[int(self.layer)].get_config()
        layer_name=layer_info['name']
        input_shape=self.model.layers[int(self.layer)].input_shape
        output_text = ('Layer Name : ' + layer_name + '\n'  
                       + 'Layer Number : ' + self.layer + '\n'
                       + 'Input Shape : ' + str(input_shape) + '\n'
                       + 'output shape :' + str(output_shape)+ '\n'
                       + 'num of feature maps :' + str(num_of_featuremaps)+ '\n'
                       + 'size of feature maps :'+ str(featuremap_size))
        self.textEdit_2.setText( output_text)

            
    def featureMaps(self):
        fpath=os.getcwd()+'/featuremap/'
        self.outputPath = (fpath + "featuremaps-layer-{}".format(self.layer) + '.png')
        output_img= cv2.imread(self.outputPath)
        output_img = cv2.resize(output_img,(550,550))
        output_profile = QtGui.QImage(self.outputPath)  
        output_profile = output_profile.scaled(550,550, aspectRatioMode=QtCore.Qt.KeepAspectRatio, 
                                             transformMode=QtCore.Qt.SmoothTransformation)
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(output_profile))
    
    def individualFeatureMap(self):
        self.map_num = self.spinBox_2.text()
        feature_maps = self.activations[0][0]
        fig=plt.figure(figsize=(16,16))
        plt.imshow(feature_maps[:,:,int(self.map_num)],cmap='gray')
        fpath=os.getcwd()+'/featuremap/'
        plt.savefig(fpath+"featuremaps-layer-{}".format(self.layer) + 'map_num-{}'.format(self.map_num)+ '.png')
        self.fPath = (fpath+"featuremaps-layer-{}".format(self.layer) + 'map_num-{}'.format(self.map_num)+ '.png')
        featuremap_img= cv2.imread(self.fPath)
        featuremap_img = cv2.resize(featuremap_img,(550,550))
        featuremap_profile = QtGui.QImage(self.fPath)  
        featuremap_profile = featuremap_profile.scaled(550,550, aspectRatioMode=QtCore.Qt.KeepAspectRatio, 
                                             transformMode=QtCore.Qt.SmoothTransformation)                       
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(featuremap_profile)) 
        
    
def main():
    
    app = QtGui.QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()


