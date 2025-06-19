import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # on doit forcer Tensorflow à désactiver Cuda pour éviter les erreurs de mémoire GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # pour éviter les messages d'avertissement de Tensorflow
import numpy as np
#{import pdb #solo per debug
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QLabel, QMessageBox
from PyQt5.uic import loadUi
from PyQt5.uic.properties import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from ela import convert_to_ela_image, convert_to_bn_image
from prediction import predict_result, find_forged_region
import tensorflow as tf


tf.keras.backend.clear_session()

#pdb.set_trace()

class MainWindow(QDialog):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("gui.ui", self)
        self.Cherche.clicked.connect(self.open_image)
        self.Essaie.clicked.connect(self.result)
        """
        if self.result=="Falsifiée":
        self.Test.clicked.connect(self.show_forged_region)
        """
        self.Quitte.clicked.connect(self.close_main_window)

    def open_image(self):
        # display original image
        self.fname = QFileDialog.getOpenFileName(
            self, "Open file", "C:/Users/FilippoStellin/OneDrive - QbitSoft/Bureau/Image_forgery_detection/Images_essai/", ("*.png, *.xmp *.jpg *.jpeg")
        )
        self.filename.setText(self.fname[0])
        pixmap = QPixmap(self.fname[0])
        self.ORIGINAL_IMAGE.setPixmap(pixmap)
        self.ORIGINAL_IMAGE.setPixmap(
            pixmap.scaled(self.ORIGINAL_IMAGE.size(), Qt.IgnoreAspectRatio)
        )
        self.ORIGINAL_IMAGE.show()

        # display ela image
        convert_to_ela_image(self.fname[0], 95) #90
        pixmap1 = QPixmap("ela_image.png")
        self.ELA_IMAGE.setPixmap(pixmap1)
        self.ELA_IMAGE.setPixmap(
            pixmap1.scaled(self.ELA_IMAGE.size(), Qt.IgnoreAspectRatio)
        )
        self.ELA_IMAGE.show()
        
        segm_test_image = Image.fromarray(np.zeros((128,128),dtype=np.int64), mode='L')  # 'L' mode for 8-bit grayscale
        segm_test_image.save("bw_image.png")
        pixmap3 = QPixmap("bw_image.png")
        self.SEGM_IMAGE.setPixmap(pixmap3)
        self.SEGM_IMAGE.setPixmap(pixmap3.scaled(self.SEGM_IMAGE.size(), Qt.IgnoreAspectRatio))
        self.SEGM_IMAGE.show()

    def result(self):
        n_pix_h=128
        n_pix_v=128
        (test_image, test_image_d, scale, prediction, confidence) = predict_result((self.fname), n_pix_h, n_pix_v)
        self.Result.setText(f"Prédiction: {prediction}\nNiveau de confiance : {confidence} %")
        
        if prediction=="Falsifiée":
            segm_image=find_forged_region(self.fname, test_image_d, n_pix_h, n_pix_v)
            #show here the forged region
            convert_to_bn_image(segm_image, n_pix_h, n_pix_v)
            
            folder_name=str(self.fname).split("'")[1].split("Images_essai")[0]+"Images_infos/"
            file_name=str(self.fname).split("/Images_essai/")[1].split(".")[0]+'_segm_image.txt'
            print("Fichier contenant l'intensité des pixels de l'image segmentée :",file_name)
            with open(folder_name+file_name, 'w') as f:
                n_px=len(segm_image)
                unique, counts = np.unique(segm_image, return_counts=True)
                f.write("Nombre de pixels :"+str(n_px)+"\n")
                f.write("Nombre de pixels originaux :"+str(counts[0])+"\n")
                if len(unique)>1:
                    f.write("Nombre de pixels modifiés :"+str(counts[1])+"\n")
                else:
                    f.write("Aucun pixel modifié trouvé !\n")
                f.write("Couleurs B/N :"+"\n")
                for i in range(n_px):
                    f.write(str(i)+"\t"+str(segm_image[i])+"\n")
            
            
            pixmap3 = QPixmap("bw_image.png")
            self.SEGM_IMAGE.setPixmap(pixmap3)
            self.SEGM_IMAGE.setPixmap(pixmap3.scaled(self.SEGM_IMAGE.size(), Qt.IgnoreAspectRatio))
            self.SEGM_IMAGE.show()
        
        pred_file_name="Prediction_results.txt"
        with open(pred_file_name, 'a') as pf:
            pf.write(str(self.fname)+"\t"+str(scale)+"\t"+str(prediction)+"\t"+str(confidence)+"\n")
          
        #return(segm_image) #aggiunto, non necessario
        
    """
    def show_forged_region()
        

    """
    def close_main_window(self):
        # quit window
        reply = QMessageBox.question(
            self,
            "Sortir", #Sortir
            "Ëtes-vous sûr de sortir ?", #Ëtes-vous sûr de sortir 
            QMessageBox.Cancel | QMessageBox.Close,
        )
        if reply == QMessageBox.Close:
            sys.exit()


def main():
    app = QApplication(sys.argv)
    Result = MainWindow()
    widget = QtWidgets.QStackedWidget()
    widget.addWidget(Result)
    widget.setFixedWidth(920) #620
    widget.setFixedHeight(560)
    widget.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
