import cv2
import numpy as np 
import imutils
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
from tensorflow.python.platform.tf_logging import error
import time

# Schnittstellen für die Kommandozeilen mit argparse definieren.
ap = argparse.ArgumentParser() # https://docs.python.org/3/library/argparse.html
ap.add_argument("-m", "--model", type=str, required=True, help="Pfad zum trainierten Tensorflow-Modell")
ap.add_argument("-c", "--cnn", type=bool, required=False, default=False, help="ist das Modell ein CNN mit quadratischem Input?")
args = vars(ap.parse_args())

#%%

# mouse callback function
def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255),thickness=10)
            pt1_x,pt1_y=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255),thickness=3)        

def pred(img, keras_model):
    if np.max(img) >1:
        # Das Bild "weichzeichnen"/Rauschen unterdrücken:
        Weichzeichnen = cv2.GaussianBlur(img, (5, 5), 0)
        # Kantenerkennung:
        Kanten = cv2.Canny(Weichzeichnen, 30, 350) # 'canny edge detection algorithm'
        # Konturerkennung:
        Konturen = cv2.findContours(Kanten.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Konturen sortieren:
        Konturen = imutils.grab_contours(Konturen)
        Konturen = sort_contours(Konturen, method="left-to-right")[0]

        for i in Konturen:
            # Rechteckigen Rahmen - in dem die Kontur i gefunden wurde - berechnen.
            (x, y, b, h) = cv2.boundingRect(i)

            # Nur Rahmen in einer bestimmten Größenordnung verwenden: b - Breite; h - Höhe.
            if (b >= 5  and h >= 20):


                # Rechteckigen Rahmen - in dem die Kontur i gefunden wurde - berechnen.
                (x, y, b, h) = cv2.boundingRect(Konturen[0])


                # Das Graustufen-Bild wiederverwenden: Das aktuelle Rechteck aus dem Bild ausschneiden und es in 'Rahmen' speichern.
                Rahmen = img[y:y + h, x:x + b]
                # Eine Schwelle definieren um eine Schwarz-Weiß-Bild 'SW' zu erzeugen.
                # Eine weiße Zahl auf einem schwarzen Hintergrund.
                SW = cv2.threshold(Rahmen, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                (H_SW, B_SW) = SW.shape

                # Größe des Rechtecks anpassen, je nachdem, welches Maß (Breite/ Höhe) größer ist.
                if B_SW > H_SW:                         # Breite ist größer als Höhe:
                    SW = imutils.resize(SW, width=22)   # Bild entlang der Breite sklaieren ~ Größenverhältnisse unverändert lassen.
                else:                                   # Höhe ist größer als Breite:
                    SW = imutils.resize(SW, height=22)  # Bild entlang der Höhe sklalieren ~ Größenverhältnisse unverändert lassen.
                # Das Rechteck ist noch kein Quadrat!
                (H_SW, B_SW) = SW.shape
                dX = int(max(0, 28 - B_SW) / 2.0)
                dY = int(max(0, 28 - H_SW) / 2.0)

                # Dem Rahmen eine 28x28 Größe auferzwingen --> Rechteck wird quadratisch:
                # Neue Pixel werden über Exptrapolation aufgefüllt (siehe https://docs.opencv.org/3.4/dc/da3/tutorial_copyMakeBorder.html)
                Quadrat = cv2.copyMakeBorder(SW, top=dY, bottom=dY,left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,value=(0, 0, 0))
                Quadrat = cv2.resize(Quadrat, (28, 28))

                # KLassifizierung unmittelbar vorbereiten:
                # Datentyp & Dimension anpassen:
                Quadrat = Quadrat.astype("float32") / 255
                Quadrat = np.expand_dims(Quadrat, axis=-1)

                if args['cnn']:
                    Vorhersage = keras_model.predict(Quadrat)
                else:
                    Vorhersage = keras_model.predict(Quadrat.reshape(-1,784))

                # Label definieren als String:
                Label = ['0','1','2','3','4','5','6','7','8','9',
                        'A','B','C','D','E','F','G','H','I','J','K','L',
                        'M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

                #for (p, (x, y, b, h)) in zip(Vorhersage, (x, y, b, h)):
                #    # Klassifizierung über Wahrscheinlichkeiten = Zuordnung anhand der höchsten Wahrscheinlichkeit:
                i = np.argmax(Vorhersage)
                Wahrscheinlichkeit = Vorhersage[0][i]
                Zuordnung = Label[i]

                # Ausgabe definieren:
                print("[INFO] {} - {:.2f}%".format(Zuordnung, Wahrscheinlichkeit * 100))
                return True, (x, y, b, h), Zuordnung
    return False, (0,0,0,0), 0


drawing = False # true if mouse is pressed
pt1_x , pt1_y = None , None

img = np.zeros((200,200), np.uint8)
cv2.namedWindow('test draw')
cv2.setMouseCallback('test draw',line_drawing)

#args = {'model':'../keras_mnist.h5', 'image':'../IMG_0305.jpg'}
keras_model = load_model(args["model"])

prediction = False
while(1):
    cv2.imshow('test draw',img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    if cv2.waitKey(1) & 0xFF == ord('s'):
        img = np.zeros((200,200), np.uint8)

    if cv2.waitKey(1) & 0xFF == ord('p'):
        erfolg, (x, y, b, h), Zuordnung = pred(img, keras_model)
        if erfolg:
            cv2.rectangle(img, (x-2, y-2), (x + b+4, y + h+4), (255), 1)
            cv2.putText(img, Zuordnung, (5,35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255), 2)
            cv2.imshow('test draw',img)
            time.sleep(1)
            cv2.waitKey(0)
            img = np.zeros((200,200), np.uint8)




cv2.destroyAllWindows()


# %%
