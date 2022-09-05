#
from email.mime import image
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

def load_data(path):
    with np.load(path, allow_pickle=True) as f:
        data, label = f['data'], f['labels']
    return data, label

def plot_number(text):
    plt.title(text)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    
def plot_numbers(x, y, daten, label):
    # create graphic with certain size
    plt.figure(figsize = (2*y,2*x))
    Label_name = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # for the axes x,y
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

def classify_image(path_to_image, path_to_model, CNN=False):
    #%%
    # Store the trained Tensorflow model locally.
    Classification = load_model(path_to_model)
    print('\n  Classify image:', path_to_image)
    # Image processing: 
    # Load input image 'image':
    Image = cv2.imread(path_to_image)
    # Convert the image to grayscale:
    Grayscale = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    # Blur the image to suppress noise:
    Blur = cv2.GaussianBlur(Grayscale, (5, 5), 0)
    # Edge detection:
    Edges = cv2.Canny(Blur, 30, 350) # 'canny edge detection algorithm'
    # Contour detection:
    Contours = cv2.findContours(Edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours:
    Contours = imutils.grab_contours(Contours)
    Contours = sort_contours(Contours, method="left-to-right")[0]

    # Initalize list - here rectangular frames from the input image come in, in which contours were found.
    rectangular_frames = []

    for i in Contours:
        # Calculate rectangular frame - in which the contour i was found.
        (x, y, b, h) = cv2.boundingRect(i)

        # Use only frames of a certain size: b - width; h - height.
        if (b >= Image.shape[1]*0.013 and b <= Image.shape[1]*0.2) and (h >= Image.shape[0]*0.3 and h <= Image.shape[1]*0.8):

            # if (b >= 20 and b <= 350) and (h >= 50 and h <= 320):
            # Reuse the grayscale image: Cut the current rectangle from the image and save it in 'Frames'.
            Frames = Grayscale[y:y + h, x:x + b]
            # Define a threshold to create a black and white 'SW' image.
            # A white figure on a black background.
            SW = cv2.threshold(Frames, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (H_SW, B_SW) = SW.shape

            # Adjust the size of the rectangle depending on which dimension (width/height) is larger.
            if B_SW > H_SW:                             # width is greater than height:
                SW = imutils.resize(SW, width=20)       # scale image along width ~ leave proportions unchanged.
            else:                                       # height is greater than width:
                SW = imutils.resize(SW, height=20)      # scale image along height ~ leave proportions unchanged.
            
            # The rectangle is not yet a square!
            (H_SW, B_SW) = SW.shape
            dX = int(max(0, 28 - B_SW) / 2.0)
            dY = int(max(0, 28 - H_SW) / 2.0)

            # Impose a 28x28 size on the frame --> Rectangle becomes square:
            # New pixels are filled via exptrapolation 
            Square = cv2.copyMakeBorder(SW, top=dY, bottom=dY,left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,value=(0, 0, 0))
            Square = cv2.resize(Square, (28, 28))
            
            # Prepare KLassification immediately:
            # Adjust data type & dimension:
            Square = Square.astype("float32") / 255
            Square = np.expand_dims(Square, axis=-1)

            # Fill list from above via for-loop (i) with: current, black & white, filled image 'square' and...
                # the coordinates & dimensions from the original image.
            rectangular_frames.append((Square, (x, y, b, h)))




    Box = [B[1] for B in rectangular_frames]
    rectangular_frames = np.array([i[0] for i in rectangular_frames], dtype="float32")
    if CNN==False:  
        rectangular_frames = rectangular_frames.reshape(-1, 784) #28x28=28^2=784, because it is square.
    Vorhersage = Classification.predict(rectangular_frames)

    # Define label as string:
    Label = "0123456789"
    Label = Label + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # Store individual strings in a list:
    Label = [l for l in Label] # Liste von [0...Z]
    
    for (p, (x, y, b, h)) in zip(Vorhersage, Box):
        # Classification via probabilities = assignment based on the highest probability:
        i = np.argmax(p)
        Wahrscheinlichkeit = p[i]
        Zuordnung = Label[i]

        # second choice
        i2 = np.argsort(p)[-2]
        Wahrscheinlichkeit2 = p[i2]
        Zuordnung2 = Label[i2]

        # Define output:
        print("[INFO] {} - {:.2f}% ---- {} - {:.2f}% \n".format(Zuordnung, Wahrscheinlichkeit * 100, Zuordnung2, Wahrscheinlichkeit2 * 100))
        cv2.rectangle(Image, (x, y), (x + b, y + h), (0, 255, 0), 2)
        cv2.putText(Image, Zuordnung, (x + int(b/2), y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    #%%
    plt.axis('off')
    plt.imshow(cv2.cvtColor(Image, cv2.COLOR_BGR2RGB))
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