import numpy as np
import cv2
from pathlib import Path

from matplotlib import pyplot as plt

import sys
if len(sys.argv) != 2:
  print ("Usage :",sys.argv[0],"detector(= orb ou kaze)")
  sys.exit(2)
if sys.argv[1].lower() == "orb":
  detector = 1
elif sys.argv[1].lower() == "kaze":
  detector = 2
else:
  print ("Usage :",sys.argv[0],"detector(= orb ou kaze)")
  sys.exit(2)
detector_name = "orb" if detector == 1 else "kaze"

#Lecture de la paire d'images (chemins relatifs al archivo)
base_dir = Path(__file__).resolve().parent
image_pairs_dir = base_dir / 'Image_Pairs'
if not (image_pairs_dir / 'torb_small1.png').exists():
  image_pairs_dir = image_pairs_dir / 'Image_Pairs'

img1_path = image_pairs_dir / 'torb_small1.png'
img2_path = image_pairs_dir / 'torb_small2.png'
img1 = cv2.imread(str(img1_path))
img2 = cv2.imread(str(img2_path))

if img1 is None or img2 is None:
  print("Erreur : impossible de lire les images.")
  print("Chemins essayés :")
  print(" -", img1_path)
  print(" -", img2_path)
  sys.exit(2)

print("Dimension de l'image 1 :",img1.shape[0],"lignes x",img1.shape[1],"colonnes")
print("Type de l'image 1 :",img1.dtype)
print("Dimension de l'image 2 :",img2.shape[0],"lignes x",img2.shape[1],"colonnes")
print("Type de l'image 2 :",img2.dtype)

#Début du calcul
t1 = cv2.getTickCount()
#Création des objets "keypoints"
if detector == 1:
  kp1 = cv2.ORB_create(nfeatures = 500,#Par défaut : 500
                       scaleFactor = 1.2,#Par défaut : 1.2
                       nlevels = 8)#Par défaut : 8
  kp2 = cv2.ORB_create(nfeatures=500,
                        scaleFactor = 1.2,
                        nlevels = 8)
  print("Détecteur : ORB")
else:
  kp1 = cv2.KAZE_create(upright = False,#Par défaut : false
    		        threshold = 0.001,#Par défaut : 0.001
  		        nOctaves = 4,#Par défaut : 4
		        nOctaveLayers = 4,#Par défaut : 4
		        diffusivity = 2)#Par défaut : 2
  kp2 = cv2.KAZE_create(upright = False,#Par défaut : false
	  	        threshold = 0.001,#Par défaut : 0.001
		        nOctaves = 4,#Par défaut : 4
		        nOctaveLayers = 4,#Par défaut : 4
		        diffusivity = 2)#Par défaut : 2
  print("Détecteur : KAZE")
#Conversion en niveau de gris
gray1 =  cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 =  cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#Détection et description des keypoints
pts1, desc1 = kp1.detectAndCompute(gray1,None)
pts2, desc2 = kp2.detectAndCompute(gray2,None)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Détection points et calcul descripteurs :",time,"s")
# Vérif des descripteurs
if desc1 is None or desc2 is None:
  print("Erreur : descripteurs non calculés (images vides ou pas de points).")
  sys.exit(2)
# Calcul de l'appariement
t1 = cv2.getTickCount()
# Paramètres de FLANN 
if detector == 1:
  # ORB -> descripteurs binaires, utiliser LSH
  FLANN_INDEX_LSH = 6
  index_params = dict(algorithm = FLANN_INDEX_LSH,
                      table_number = 6,
                      key_size = 12,
                      multi_probe_level = 1)
  search_params = dict(checks=50)
  flann = cv2.FlannBasedMatcher(index_params, search_params)
else:
  # KAZE -> descripteurs float, utiliser KD-Tree
  FLANN_INDEX_KDTREE = 0
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  search_params = dict(checks=50)
  flann = cv2.FlannBasedMatcher(index_params, search_params)
  # OpenCV exige float32 pour KD-Tree
  if desc1.dtype != np.float32:
    desc1 = np.float32(desc1)
  if desc2.dtype != np.float32:
    desc2 = np.float32(desc2)

matches = flann.knnMatch(desc1,desc2,k=2)
# Application du ratio test
good = []
for m,n in matches:
  if m.distance < 0.7*n.distance:
    good.append([m])
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Calcul de l'appariement :",time,"s")

# Affichage
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   flags = 0)

# Affichage des appariements qui respectent le ratio test
img3 = cv2.drawMatchesKnn(gray1,pts1,gray2,pts2,good,None,**draw_params)

Nb_ok = len(good)
output_dir = base_dir / 'docs' / 'rappport' / 'imgs' / 'descriptors'
output_dir.mkdir(parents=True, exist_ok=True)
out_path = output_dir / f"{Path(__file__).stem}_{detector_name}.png"
cv2.imwrite(str(out_path), img3)
print("Imagen guardada:", out_path)
plt.imshow(img3),plt.title('%i appariements OK'%Nb_ok)
plt.show()
