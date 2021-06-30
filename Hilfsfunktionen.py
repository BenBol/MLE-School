#
import matplotlib.pyplot as plt
import numpy as np
import os, random
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import imutils
import cv2

# Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

plt.rcParams['image.cmap'] = 'binary'

def lade_daten(path):
    with np.load(path, allow_pickle=True) as f:
        data, label = f['data'], f['labels']
    return data, label

def plotte_zahl(text):
    plt.title(text)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    
def plotte_Zahlen(x, y, daten, label):
    # erstelle Grafik mit bestimmter Größe
    plt.figure(figsize = (2*y,2*x))
    Label_name = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # für die achsen x,y
    for i in range(x*y):
        plt.subplot(x,y,i+1)
        random_index = random.randrange(len(daten))
        plt.imshow(daten[random_index])
        plt.title("Zahl: {}".format(Label_name[label[random_index][0]]))
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    plt.show()
    return random_index

def klassifiziere_bild(path_to_image, path_to_model, CNN=False):
    #%%
    # Das trainierte Tensorflow-Modell lokal speichern.
    Klassifizierung = load_model(path_to_model)
    print('\n  Klassifiziere Bild:', path_to_image)
    # Bildverarbeitung: 
    # Eingangsbild 'image' laden:
    Bild = cv2.imread(path_to_image)
    # Das Bild zu Graustufen konvertieren:
    Graustufen = cv2.cvtColor(Bild, cv2.COLOR_BGR2GRAY)
    # Das Bild "weichzeichnen"/Rauschen unterdrücken:
    Weichzeichnen = cv2.GaussianBlur(Graustufen, (5, 5), 0)
    # Kantenerkennung:
    Kanten = cv2.Canny(Weichzeichnen, 30, 350) # 'canny edge detection algorithm'
    # Konturerkennung:
    Konturen = cv2.findContours(Kanten.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Konturen sortieren:
    Konturen = imutils.grab_contours(Konturen)
    Konturen = sort_contours(Konturen, method="left-to-right")[0]

    # Liste initalisieren - hier kommen rechteckige Rahmen aus dem Eingangsbild rein, in denen Konturen gefunden wurden.
    RRahmen = []

    for i in Konturen:
        # Rechteckigen Rahmen - in dem die Kontur i gefunden wurde - berechnen.
        (x, y, b, h) = cv2.boundingRect(i)

        # Nur Rahmen in einer bestimmten Größenordnung verwenden: b - Breite; h - Höhe.
        if (b >= Bild.shape[1]*0.013 and b <= Bild.shape[1]*0.2) and (h >= Bild.shape[0]*0.3 and h <= Bild.shape[1]*0.8):

            # if (b >= 20 and b <= 350) and (h >= 50 and h <= 320):
            # Das Graustufen-Bild wiederverwenden: Das aktuelle Rechteck aus dem Bild ausschneiden und es in 'Rahmen' speichern.
            Rahmen = Graustufen[y:y + h, x:x + b]
            # Eine Schwelle definieren um eine Schwarz-Weiß-Bild 'SW' zu erzeugen.
            # Eine weiße Zahl auf einem schwarzen Hintergrund.
            SW = cv2.threshold(Rahmen, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (H_SW, B_SW) = SW.shape

            # Größe des Rechtecks anpassen, je nachdem, welches Maß (Breite/ Höhe) größer ist.
            if B_SW > H_SW:                         # Breite ist größer als Höhe:
                SW = imutils.resize(SW, width=20)   # Bild entlang der Breite sklaieren ~ Größenverhältnisse unverändert lassen.
            else:                                   # Höhe ist größer als Breite:
                SW = imutils.resize(SW, height=20)  # Bild entlang der Höhe sklalieren ~ Größenverhältnisse unverändert lassen.
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

            # Liste von oben via for-Schleife (i) auffüllen mit: aktuellen, schwarz-weißen, aufgefüllten Bild 'Quadrat' und...
                # den KOOD & Maßen vom ursprünglichen Bild.
            RRahmen.append((Quadrat, (x, y, b, h)))




    Box = [B[1] for B in RRahmen]
    RRahmen = np.array([i[0] for i in RRahmen], dtype="float32")
    if CNN==False:  
        RRahmen = RRahmen.reshape(-1, 784) #28x28=28^2=784, weil es quadratisch ist.
    Vorhersage = Klassifizierung.predict(RRahmen)

    # Label definieren als String:
    Label = "0123456789"
    Label = Label + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # Einzelne Strings in einer Liste abspeichern:
    Label = [l for l in Label] # Liste von [0...Z]
    
    for (p, (x, y, b, h)) in zip(Vorhersage, Box):
        # Klassifizierung über Wahrscheinlichkeiten = Zuordnung anhand der höchsten Wahrscheinlichkeit:
        i = np.argmax(p)
        Wahrscheinlichkeit = p[i]
        Zuordnung = Label[i]

        # zweite Wahl
        i2 = np.argsort(p)[-2]
        Wahrscheinlichkeit2 = p[i2]
        Zuordnung2 = Label[i2]

        # Ausgabe definieren:
        print("[INFO] {} - {:.2f}% ---- {} - {:.2f}% \n".format(Zuordnung, Wahrscheinlichkeit * 100, Zuordnung2, Wahrscheinlichkeit2 * 100))
        cv2.rectangle(Bild, (x, y), (x + b, y + h), (0, 255, 0), 2)
        cv2.putText(Bild, Zuordnung, (x + int(b/2), y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    #%%
    plt.axis('off')
    plt.imshow(cv2.cvtColor(Bild, cv2.COLOR_BGR2RGB))
    plt.savefig(path_to_image[:-4]+'_'+path_to_model.split(os.sep)[-1][:-3]+'.jpg', bbox_inches='tight', pad_inches=0.1)

    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes=[],
                          normalize=False,
                          plot_title=[],
                          cmap=plt.cm.Blues, ax = []):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    

    # Compute confusion matrix
    cm = confusion_matrix(y_true, np.array(y_pred).flatten())
    # Only use the labels that appear in the data
    if classes==[]:
        classes = unique_labels(y_true, y_pred)
        
    #classes=['a', 'b', 'c']    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]*100

    
    if ax == []:
        fig, ax = plt.subplots(figsize  = (9,8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Accuracy / %')
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           
           ylabel='True State',
           xlabel='Predicted State')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),  ha="center",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.0f' if normalize else 'd'
    thresh = ((cm.max()-cm.min()) / 2.)+cm.min()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    if plot_title != []:
        plt.title(plot_title)
    return ax