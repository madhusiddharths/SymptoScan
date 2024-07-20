# SymptoScan
A disease classification model!!

This model aims to classify 4 closely related diseases - cold, flu, covid, allergy. Since a good decision boundary is non existant(due to similar features), a conventional multiclass classifier fails to classify properly. So, this approaches introduces the concept of stacking classifier where individual models are built for each disease (1 vs all classifier) and finally an meta-classifier is built on top of those output.
