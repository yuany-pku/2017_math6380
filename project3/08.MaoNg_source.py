#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.applications import vgg16
from keras.models import Model
from keras import backend as K
from sklearn import preprocessing
from sklearn.metrics import pairwise
from sklearn import decomposition
import os
import cv2
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def load_model(incl_top=False): return vgg16.VGG16(weights='imagenet', include_top=incl_top)
#base_model = load_model()
#entire_model = load_model(True)

def load_img_file(path):
    f_name = os.listdir(path)
    f_name.sort(key=lambda x:int(os.path.splitext(x)[0]))
    return [os.path.join(path,f) for f in f_name]

def load_img_cv2(path, target_size=None):
    img = cv2.imread(path,1)
    if target_size:
        hw_tuple = (target_size[1], target_size[0])
        if img.size != hw_tuple:
            img = cv2.resize(img, hw_tuple)
    return img

def preprocess_image(image_path, output_size=(224,224)):
    img = load_img_cv2(image_path, target_size=output_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img

def gram_matrix(x):
    assert x.ndim == 4
    if K.image_dim_ordering() == 'th':
        features = batch_flatten(x)
    else:
        features = batch_flatten(x.transpose(0,3,1,2))
    gram = np.dot(features, features.T)
    return gram

def batch_flatten(x): return x.reshape(np.shape(x)[1], -1)

def style_feat(image):
    model = Model(input=entire_model.input, output=entire_model.get_layer('block5_conv1').output)
    return gram_matrix(model.predict(image))

def fc_feat(image):
    model = Model(input=entire_model.input, output=entire_model.get_layer('fc2').output)
    return model.predict(image)
    
def sc_encoding(image_path):
    #width, height = load_img(image_path).size
    image = preprocess_image(image_path) 
    #image = preprocess_image(image_path, output_size=(height, width))
    C_feat = fc_feat(image)
    S_feat = style_feat(image)
    return S_feat.reshape(1,-1),C_feat

def dim_reduction(x, alg='pca', n_comp=2048):
    if alg == 'pca':
        return decomposition.PCA(n_components=n_comp).fit_transform(x)
    else:
        return decomposition.SparsePCA(n_components=n_comp).fit_transform(x)

def save_feat(f_list):
    c_matrix = np.zeros((28,4096))
    s_matrix = np.zeros((28,262144))
    ind = 0
    for f in f_list:
        s_f, c_f = sc_encoding(f)
        c_matrix[ind] = c_f
        s_matrix[ind] = s_f
        print('Num: '+str(ind))
        ind = ind+1
    np.save('c_matrix.npy',c_matrix)
    np.save('s_matrix.npy',s_matrix)
    # 1=true 0=fake 2=not sure
    label = np.array([2,1,1,1,1,1,
                      2,1,1,2,0,0,
                      0,0,0,0,0,0,
                      0,2,1,1,2,1,
                      2,2,1,1])
    np.save('label.npy',label)

def load_feat(f_name):
    f_matrix = np.load(f_name)
    label = np.load('label.npy')
    return f_matrix, label

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def classifier(data, label):
    binary_data = np.zeros((21,data.shape[1]))
    notsure_data = np.zeros((7,data.shape[1]))
    new_label = []
    ind = 0
    ind_1 = 0
    for i, l in enumerate(label):
        if l == 2:
            notsure_data[ind_1] = data[i]
            ind_1 = ind_1+1
            continue
        binary_data[ind] = data[i]
        new_label.append(l)
        ind = ind+1
    binary_data = preprocessing.normalize(binary_data)
    #pca = decomposition.PCA(n_components=256)
    #binary_data = pca.fit_transform(binary_data)
    new_label = np.array(new_label)
    X_train, X_test, y_train, y_test = train_test_split(binary_data, new_label, test_size=.3,
                                                    random_state=np.random.RandomState(0))
    '''X_train = np.vstack((binary_data[0:7,:],binary_data[7:12,:]))
    y_train = np.array([1,1,1,1,1,1,1,
                               0,0,0,0,0])
    X_test = np.vstack((binary_data[16:21,:],binary_data[12:16,:]))
    y_test = np.array([1,1,1,1,1,
                              0,0,0,0])'''
    clf_1 = MLPClassifier()
    clf_1.fit(X_train, y_train)
    pre_1 = clf_1.predict(X_test)
    p_1 = clf_1.predict_proba(notsure_data)
    pnotsure_1 = clf_1.predict(notsure_data)
    #t_score_1 = clf_1.score(X_test,y_test)
    #t_score_1 = clf_1.decision_function(X_test)
    print(metrics.classification_report(y_test, pre_1, target_names=['Fake','True']))
    
    clf_2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                         algorithm="SAMME",
                         n_estimators=25)
    clf_2.fit(X_train, y_train)
    t_score_2 = clf_2.decision_function(X_test)
    pre_2 = clf_2.predict(X_test)
    print(metrics.classification_report(y_test, pre_2, target_names=['Fake','True']))
    C_range = 2. ** np.arange(-5, 15)
    gamma_range = 2. ** np.arange(-15, 3)
    param_grid = dict(gamma=gamma_range, C=C_range)
    clf_3 = GridSearchCV(SVC(kernel='rbf',probability=True), param_grid)
    clf_3.fit(X_train, y_train)
    t_score_3 = clf_3.decision_function(X_test)
    pre_3= clf_3.predict(X_test)
    draw_pre_recall(t_score_2,t_score_3 , y_test)
    '''scores = clf.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()'''
    
    print(metrics.classification_report(y_test, pre_3, target_names=['Fake','True']))
    '''loo = LeaveOneOut()
    acc = []
    for train_index, test_index in loo.split(binary_data):
        X_train, X_test = binary_data[train_index], binary_data[test_index]
        y_train, y_test = new_label[train_index], new_label[test_index]
        clf.fit(X_train, y_train)
        pre = clf.predict(X_test)
        pre_tr = clf.predict(X_train)
        print('===================================')
        print('Real: ',y_test)
        print('Pre: ',pre)
        print('Real_tr: ', y_train)
        print('Pre_tr: ',pre_tr)
        print((sum(y_train==pre_tr)+sum(y_test==pre))/21.0)
        acc.append((sum(y_train==pre_tr)+sum(y_test==pre))/21.0)
    print('Summary: ', sum(acc)/21.0)'''
    p_2 = clf_2.predict_proba(notsure_data)
    pnotsure_2 = clf_2.predict(notsure_data)
    p_3 = clf_3.predict_proba(notsure_data)
    pnotsure_3 = clf_3.predict(notsure_data)
    return (p_1, p_2,p_3, pnotsure_1,pnotsure_2,pnotsure_3)

def draw_pre_recall(y_score_2,y_score_3 , y_test):
    # Compute Precision-Recall and plot curve
    #precision_1, recall_1, thresholds_1 = precision_recall_curve(y_test, y_score_1)
    precision_2, recall_2, thresholds_2 = precision_recall_curve(y_test, y_score_2)
    precision_3, recall_3, thresholds_3 = precision_recall_curve(y_test, y_score_3)
    plt.clf()
    #plt.plot(recall_1, precision_1, label='MLP')
    plt.plot(recall_2, precision_2, label='AdaBoost')
    plt.plot(recall_3, precision_3, label='SVM')
    plt.title('Precision and Recall Curve (Cross Val=0.3)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()
  
if __name__ == "__main__":
    #f_list = load_img_file('img/')
    #save_feat(f_list)
    f_matrix, label = load_feat('c_matrix.npy')
    a = classifier(f_matrix, label)