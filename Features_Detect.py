"""
python Features_Detect.py orb
python Features_Detect.py kaze
python Features_Detect.py orb -stats
python Features_Detect.py kaze -stats 100
"""


import argparse
import os

import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def parse_args():
  parser = argparse.ArgumentParser(
      description="Détection des points d'intérêt avec ORB ou KAZE."
  )
  parser.add_argument("detector", choices=["orb", "kaze"], help="Détecteur à utiliser.")
  parser.add_argument(
      "-stats",
      nargs="?",
      const=50,
      type=int,
      default=None,
      metavar="N",
      help="Répéter N fois la détection et afficher moyenne, variance, écart-type (N=50 par défaut).",
  )
  args = parser.parse_args()
  if args.stats is not None and args.stats <= 0:
    parser.error("N doit être un entier strictement positif.")
  return args


args = parse_args()
detector = 1 if args.detector.lower() == "orb" else 2

#Lecture de la paire d'images
img1 = cv2.imread('Image_Pairs/Image_Pairs/torb_small1.png')
print("Dimension de l'image 1 :",img1.shape[0],"lignes x",img1.shape[1],"colonnes")
print("Type de l'image 1 :",img1.dtype)
img2 = cv2.imread('Image_Pairs/Image_Pairs/torb_small2.png')
print("Dimension de l'image 2 :",img2.shape[0],"lignes x",img2.shape[1],"colonnes")
print("Type de l'image 2 :",img2.dtype)

def detect_points_with_timing(img1, img2, detector):
  t1 = cv2.getTickCount()
  if detector == 1:
    kp1 = cv2.ORB_create(nfeatures = 250,#Par défaut : 500
                         scaleFactor = 2,#Par défaut : 1.2
                         nlevels = 3)#Par défaut : 8
    kp2 = cv2.ORB_create(nfeatures=250,
                         scaleFactor = 2,
                         nlevels = 3)
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

  gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
  gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
  pts1 = kp1.detect(gray1,None)
  pts2 = kp2.detect(gray2,None)
  t2 = cv2.getTickCount()
  elapsed_time = (t2 - t1)/ cv2.getTickFrequency()
  return gray1, gray2, pts1, pts2, elapsed_time


print("Détecteur : ORB" if detector == 1 else "Détecteur : KAZE")
gray1, gray2, pts1, pts2, time = detect_points_with_timing(img1, img2, detector)
print("Détection des points d'intérêt :",time,"s")

if args.stats is not None:
  times = [time]
  for _ in range(args.stats - 1):
    _, _, _, _, elapsed_time = detect_points_with_timing(img1, img2, detector)
    times.append(elapsed_time)
  times = np.asarray(times, dtype=np.float64)
  print(f"Statistiques sur {args.stats} exécutions :")
  print(f"  Moyenne : {times.mean():.6f} s")
  print(f"  Variance : {times.var():.6e} s^2")
  print(f"  Déviation standard : {times.std():.6f} s")

#Affichage des keypoints
img1 = cv2.drawKeypoints(gray1, pts1, None, flags=4)
# flags définit le niveau d'information sur les points d'intérêt
# 0 : position seule ; 4 : position + échelle + direction
img2 = cv2.drawKeypoints(gray2, pts2, None, flags=4)

plt.subplot(121)
plt.imshow(img1)
plt.title('Image n°1')

plt.subplot(122)
plt.imshow(img2)
plt.title('Image n°2')

plt.show()
method_name = "ORB" if detector == 1 else "KAZE"
save_dir = os.path.join('results', 'method_features', method_name)
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'detected_keypoints.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Résultat sauvegardé dans {save_path}")
