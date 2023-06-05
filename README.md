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

file: analyse/03_18_exp_noise/analyse_exp.ipynb

This experiments add noise on the observation trajectory of **corupted** scenes (with the s-attack method). We then analyse the resulting prediction, in terms of colision.

### **Certification of trajectories** :

file: analyse/03_25_zero_col_certif/my_analyze.ipynb

This experiment has for goal to certify that a scene is collision free. With a maximum noise level, we can certify that the output of the "smoothed classifier" will be "no collision", for a particular scene.

### **Fde/ade under noise** :

file: analyse/04_03_f_ade_zero_col_cert/analyse.ipynb

This experiments shows the effect on the final predicted trajectory when noise is added on the observation.

### **WRONG : Zero colision predictor** :

file: analyse/04_07_all_data/analyse.ipynb

This experiment is wrong and was latter redone. It has for goal to predict a colision-free scene, even if the original preditions would have had a colision.


### **WRONG : Zero colision predictor v2** :

file: analyse/04_18_really_all_data_z_col_ade/analyse.ipynb

This experiment is wrong and was latter redone. It has for goal to predict a colision-free scene, even if the original preditions would have had a colision.

### **Analyse of data, colision rate and ade/fde**

file: analyse/04_24_secret_v2/analyse_04_24.ipynb

Having seen that the 2 previous experiements did not showcase the correct numbers, I was wondering if something was wrong with the preprocessing of the scenes or the prediction pipeline. In this file, I tried to find those problems.

### **Analyse of data, IA crowd check**

file: analyse/04_25_no_noise_clean/analyse_04_25.ipynb

In this experiment, I am doing the last data check. I try many combinaison of preprocessing options, to see wich one coresponds to the correct numbers. I also did on IA crowd submission to verify the numbers. In this file, I explain that there exists another definition for colision, and that the number I was trying to get corespond to a specific type of scenes.

### **RIGHT : Zero colision predictor v3** :

file : analyse/04_26_redo_0_col_pred/analyse_04_26.ipynb

Now that the corect way of handling NaN values is understood, we can repeat the experiment of the zero-predictor. It has for goal to predict a colision-free scene, even if the original preditions would have had a colision.

### ** Bounded trajectory regression ** :

file : analyse/05_08_bounds/analyse_08_may.ipynb

This experimement is the main topic of my project. Given a maximal perturbation radius, we show that each coordinate of a predicted trajectory can be bounded. The theory behind this experiement is explained in the following section. 
We tested different type of functions to "sumarize" the 100 noisy trajectories drawn: the mean and 2 type of median. Finally, a diffusion denoiser was also implemented.

## explain


## Bibliography

[1] Chiang, P. Y., Curry, M., Abdelkader, A., Kumar, A., Dickerson, J., & Goldstein, T. (2020). Detection as regression: Certified object detection with median smoothing. Advances in Neural Information Processing Systems, 33, 1275-1286.

[2] Salman, H., Li, J., Razenshteyn, I., Zhang, P., Zhang, H., Bubeck, S., & Yang, G. (2019). Provably robust deep learning via adversarially trained smoothed classifiers. Advances in Neural Information Processing Systems, 32.

[3] Cohen, J., Rosenfeld, E., & Kolter, Z. (2019, May). Certified adversarial robustness via randomized smoothing. In international conference on machine learning (pp. 1310-1320). PMLR.



