# Certified trajectory prediction

## Instalation (still need to check)

```
cd blabla...
pip install .
```
carefull pip install -e . doesnt work (dont allow relative import ???)

can run the the file from saeed + hossein : 
on windows
```
.\lrun.bat 
```
or on linux
```
bash lrun.sh
```

can run the experience with :
```
.\run_exp.bat
```

## Introduction 

This readme file will introduce the reader to the work I've done with my project. The experiments I've done will be briefly introduce here. More informations about those are contained in the .ipynb file.

## Experiments (chronological)

### **Noise on observations** :

file : analyse/03_18_exp_noise/analyse_exp.ipynb

This experiments add noise on the observation trajectory of **corupted** scenes (with the s-attack method). We then analyse the resulting prediction, in terms of colision.

### **Certification of trajectories** :

file : analyse/03_25_zero_col_certif/my_analyze.ipynb

This experiment has for goal to certify that a scene is collision free. With a maximum noise level, we can certify that the output of the "smoothed classifier" will be "no collision", for a particular scene.

### **Fde/ade under noise** :

file : analyse/04_03_f_ade_zero_col_cert/analyse.ipynb

This experiments shows the effect on the final predicted trajectory when noise is added on the observation.

### **WRONG : Zero colision predictor** :

file : analyse/04_07_all_data/analyse.ipynb

This experiment is wrong and was latter redone. It has for goal to predict a colision-free trajectory, even if the original predition would have had a colision.


### **WRONG : Zero colision predictor v2** :

file : analyse/04_18_really_all_data_z_col_ade/analyse.ipynb

This experiment is wrong and was latter redone. It has for goal to predict a colision-free trajectory, even if the original predition would have had a colision.

### **Analyse of data, colision rate and ade/fde**

file : analyse/04_24_secret_v2/analyse_04_24.ipynb

Having seen that the 2 previous experiements did not showcase the correct numbers, I was wondering if something was wrong with the preprocessing of the scenes or the prediction pipeline. In this file, I tried to find those problems.

