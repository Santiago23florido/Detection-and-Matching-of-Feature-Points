import numpy as np
import cv2

from matplotlib import pyplot as plt

"""
- Loads a gray image
- Harris calculation
- Local max's extraction with morphological dilation
- Sample original image, response map and interest points
"""

#---------------------------------------------
# 1. Lecture image et initialisation
#---------------------------------------------

#Lecture image en niveau de gris et conversion en float64
#   In float64 to avoid overflow during Harris calculation, avoid saturation
img=np.float64(cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_GRAYSCALE))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")
print("Type de l'image :",img.dtype)

#----------------------------------------------------------
# 1.Calcul et temporisation de la fonction d'intérêt de Harris
#----------------------------------------------------------

#Début du calcul
t1 = cv2.getTickCount() # Start time

# Copy image and add borders as a extended frame 
#   with boreder replication (but it does nothing here as there are zeros)
Theta = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)

# TODO Mettre ici le calcul de la fonction d'intérêt de Harris

    #   Calcul des gradients par différences finies

    #   Produits des gradients

    #   Somme dans une fenêtre (filtre moyenneur)

    #   Calcul de la fonction de Harris, theta


#----------------------------------------------------------
# 2. Calcul des maxima locaux et seuillage + temporisation
#----------------------------------------------------------

Theta_maxloc = cv2.copyMakeBorder(                  # Copy of image to work on
    Theta,0,0,0,0,cv2.BORDER_REPLICATE
    ) 

d_maxloc = 3                                        # Size of neighborhood for local maxima
seuil_relatif = 0.01                                # Relative threshold
se = np.ones((d_maxloc,d_maxloc),np.uint8)          # Structuring element for dilation. matrix of ones

Theta_dil = cv2.dilate(Theta, se)                   # Dilation of Harris response. New image where each point 
                                                    #   is the maximum in its neighborhood, defined by se


Theta_maxloc[Theta < Theta_dil] = 0.0               # Suppression des non-maxima-locaux. Where theta is lower than dilated image,
                                                    #       set to zero. There is no maximum there.
Theta_maxloc[Theta <                                # On néglige également les valeurs trop faibles
             seuil_relatif*Theta.max()] = 0.0       

t2 = cv2.getTickCount()                             # End time
time = (t2 - t1)/ cv2.getTickFrequency()            # Time in seconds

#----------------------------------------------------------
# 3. Affichage des résultats
#----------------------------------------------------------

print("Mon calcul des points de Harris :",time,"s")
print("Nombre de cycles par pixel :",(t2 - t1)/(h*w),"cpp")

plt.subplot(131)
plt.imshow(img,cmap = 'gray')
plt.title('Image originale')

plt.subplot(132)
plt.imshow(Theta,cmap = 'gray')
plt.title('Fonction de Harris')

se_croix = np.uint8([[1, 0, 0, 0, 1],
[0, 1, 0, 1, 0],[0, 0, 1, 0, 0],
[0, 1, 0, 1, 0],[1, 0, 0, 0, 1]])
Theta_ml_dil = cv2.dilate(Theta_maxloc,se_croix)
#Relecture image pour affichage couleur
Img_pts=cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_COLOR)
(h,w,c) = Img_pts.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes x",c,"canaux")
print("Type de l'image :",Img_pts.dtype)
#On affiche les points (croix) en rouge
Img_pts[Theta_ml_dil > 0] = [255,0,0]
plt.subplot(133)
plt.imshow(Img_pts)
plt.title('Points de Harris')

plt.show()
