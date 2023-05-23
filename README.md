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







