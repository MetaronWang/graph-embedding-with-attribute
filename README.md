**It's just a messy version so far, which is the initial one. I will refactor this project later on.**

#  Dependence
*   Tensorflow=1.14
*   Numpy=1.19.2
*   gensim=3.8.3
*   sklearn=0.23.2

# How to Run
You just need to run the scripts below in the project root folder.

If you don't want to run the MLP classifier, you don't need to install the TensorFlow package.

## Do the pre-training
```script
python node2vec_walk.py
```

## Do the classification
### For SVM、Logistic Regression、NaiveBayes、Decision Tree、Random Forest
```script
python node_classifier.py
```

### For MLP
```script
python nn_classifier.py
```

