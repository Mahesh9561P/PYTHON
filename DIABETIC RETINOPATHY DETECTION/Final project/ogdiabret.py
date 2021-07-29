import time
import gc
from statistics import mean
from os import walk
from scipy import misc
from PIL import Image
from skimage import exposure
from sklearn import svm
import scipy
from math import sqrt,pi
from numpy import exp
from matplotlib import pyplot as plt
import numpy as np
import glob
import matplotlib.pyplot as pltss
import cv2
from matplotlib import cm
import pandas as pd
from math import pi,sqrt
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st  

imm_kmean=[]
scaler_means=[]
y_result=[]
path, dirs, files = next(walk("/home/mahesh/Mahesh/Python/BEproject/diaretdb1/diaretdb1v1/diaretdb1v1v1/resources/images/db1fundusimages"))
file_count = len(files)
start_time = time.time()


def Image_Processing():
    print("STARTED WITH IMAGE PROCESSING....")

    global start_time
    immatrix=[]
    im_unpre = []

    for i in range(1,file_count):
        img_pt = r'/home/mahesh/Mahesh/Python/BEproject/diaretdb1/diaretdb1v1/diaretdb1v1v1/resources/images/db1fundusimages/image'
        if i < 10:
            img_pt = img_pt + "00" + str(i) + ".png"
        else:
            img_pt = img_pt + "0" + str(i)+ ".png"
        
        img = cv2.imread(img_pt)
        #im_unpre.append(np.array(img).flatten())
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(img_gray)
        immatrix.append(np.array(equ).flatten())
        #res = np.hstack((img_gray,equ))

    imm_dwt = []
    for equ in immatrix:
        equ = equ.reshape((1152,1500))
        coeffs = pywt.dwt2(equ, 'haar')
        equ2 = pywt.idwt2(coeffs, 'haar')
        imm_dwt.append(np.array(equ2).flatten())


    def _filter_kernel_mf_fdog(L, sigma, t = 3, mf = True):
        dim_y = int(L)
        dim_x = 2 * int(t * sigma)
        arr = np.zeros((dim_y, dim_x), 'f')
        
        ctr_x = dim_x / 2 
        ctr_y = int(dim_y / 2.)

        # an un-natural way to set elements of the array
        # to their x coordinate. 
        # x's are actually columns, so the first dimension of the iterator is used
        it = np.nditer(arr, flags=['multi_index'])
        while not it.finished:
            arr[it.multi_index] = it.multi_index[1] - ctr_x
            it.iternext()

        two_sigma_sq = 2 * sigma * sigma
        sqrt_w_pi_sigma = 1. / (sqrt(2 * pi) * sigma)
        if not mf:
            sqrt_w_pi_sigma = sqrt_w_pi_sigma / sigma ** 2

        #@vectorize(['float32(float32)'], target='cpu')
        def k_fun(x):
            return sqrt_w_pi_sigma * exp(-x * x / two_sigma_sq)

        #@vectorize(['float32(float32)'], target='cpu')
        def k_fun_derivative(x):
            return -x * sqrt_w_pi_sigma * exp(-x * x / two_sigma_sq)

        if mf:
            kernel = k_fun(arr)
            kernel = kernel - kernel.mean()
        else:
            kernel = k_fun_derivative(arr)
        # return the "convolution" kernel for filter2D
        return cv2.flip(kernel, -1) 

    def gaussian_matched_filter_kernel(L, sigma, t = 3):
        '''
        K =  1/(sqrt(2 * pi) * sigma ) * exp(-x^2/2sigma^2), |y| <= L/2, |x| < s * t
        '''
        return _filter_kernel_mf_fdog(L, sigma, t, True)

    #Creating a matched filter bank using the kernel generated from the above functions
    def createMatchedFilterBank(K, n = 12):
        rotate = 180 / n
        center = (K.shape[1] / 2, K.shape[0] / 2)
        cur_rot = 0
        kernels = [K]

        for i in range(1, n):
            cur_rot += rotate
            r_mat = cv2.getRotationMatrix2D(center, cur_rot, 1)
            k = cv2.warpAffine(K, r_mat, (K.shape[1], K.shape[0]))
            kernels.append(k)

        return kernels

    #Given a filter bank, apply them and record maximum response
    def applyFilters(im, kernels):

        images = np.array([cv2.filter2D(im, -1, k) for k in kernels])
        return np.max(images, 0)

    gf = gaussian_matched_filter_kernel(20, 5)
    bank_gf = createMatchedFilterBank(gf, 4)

    imm_gauss = []
    for equ2 in imm_dwt:
        equ2 = equ2.reshape((1152,1500))
        equ3 = applyFilters(equ2,bank_gf)
        imm_gauss.append(np.array(equ3).flatten())

    def createMatchedFilterBank():
        filters = []
        ksize = 31
        for theta in np.arange(0, np.pi, np.pi / 16):
            kern = cv2.getGaborKernel((ksize, ksize), 6, theta,12, 0.37, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
        return filters

    def applyFilters(im, kernels):
        images = np.array([cv2.filter2D(im, -1, k) for k in kernels])
        return np.max(images, 0)

    bank_gf = createMatchedFilterBank()
    #equx=equ3
    #equ3 = applyFilters(equ2,bank_gf)
    imm_gauss2 = []
    for equ2 in imm_dwt:
        equ2 = equ2.reshape((1152,1500))
        equ3 = applyFilters(equ2,bank_gf)
        imm_gauss2.append(np.array(equ3).flatten())

    e_ = equ3
    e_=e_.reshape((-1,3))
    img = equ3
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)

    k=cv2.KMEANS_PP_CENTERS

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,k)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    for equ3 in imm_gauss2:
        img = equ3.reshape((1152,1500))
        Z = img.reshape((-1,3))

        # convert to np.float32
        Z = np.float32(Z)

        k=cv2.KMEANS_PP_CENTERS

        global imm_kmean

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,k)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        imm_kmean.append(np.array(res2).flatten())   
    del imm_dwt
    del imm_gauss
    del imm_gauss2
    del immatrix
    del im_unpre
    
    gc.collect() 
    print("FINISHED WITH IMAGE PROCESSING...") 
    print("--- %s seconds to finish image processing func---" % (time.time() - start_time))   
#-------------------------------------PREPROCESSING----------------------------------------------------

def Preprocessing():
    global imm_kmean,start_time,file_count
    print("STARTED WITH PREPROCESSING...")
    '''scaler=StandardScaler()
    imm_kmean=scaler.fit_transform(imm_kmean)
'''
    total_mean=np.array(imm_kmean).mean()
    global scaler_means
    for i in range(file_count-1):
        scaler_means.append(imm_kmean[i].mean())

    global y_result
    for i in range(file_count-1):
        if imm_kmean[i].mean()>total_mean:
            y_result.append(1) 
        else :
            y_result.append(0)
    print("DONE WITH PREPROCESSING...")
    print("--- %s seconds to finish pre-processing func---" % (time.time() - start_time))

#------------------------------------------MACHINE LEARNING-------------------------------------------------

def Model_KNN(test_image,test_result):
    global start_time,y_result
    print("STARTED WITH MACHINE LEARNING...")
    reg=KNeighborsClassifier(n_neighbors=5)
    global scaler_means
    #data_train,data_test,label_train,label_test=train_test_split(scaler_means.reshape(-1,1),y_result,test_size=0.2,random_state=1)

    reg.fit(scaler_means,y_result)
    output=reg.predict(test_image)
    #print("Predicted ",output)
    #print("Actual    ",label_train)
    if output[0]==1 and test_result==1:
        return 1
    else:
        return 0
    '''accuracy=accuracy_score(test_image,output)
    print("DONE WITH MACHINE LEARNING")
    print("ACCURACY OF KNN",accuracy*100)'''
    print("--- %s seconds to finish machine learning func---" % (time.time() - start_time))
    
    #ERROR_VS_K(data_train,label_train,data_test,label_test)

    
#---------------------------------------VISUALIZATION OF ERROR_VS_K-----------------------------------------------
'''def ERROR_VS_K(data_train,label_train,data_test,label_test):
    error_rate = []
    for i in range(1,40):
        knn=KNeighborsClassifier(n_neighbors=i)
        knn.fit(data_train,label_train)
        pred_i= knn.predict(data_test)
        error_rate.append(np.mean(pred_i != label_test))

    plt.figure(figsize=(10,6))
    plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', 
            marker='o',markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))
    #ACCURACY_VS_K(data_train,label_train,data_test,label_test)
#--------------------------------------VISUALIZATION OF ACCURACY_VS_K------------------------------
def ACCURACY_VS_K(data_train,label_train,data_test,label_test):
    acc = []
    # Will take some time
    from sklearn import metrics
    for i in range(1,40):
        neigh = KNeighborsClassifier(n_neighbors = i).fit(data_train,label_train)
        yhat = neigh.predict(data_test)
        acc.append(metrics.accuracy_score(label_test, yhat))
        
    plt.figure(figsize=(10,6))
    plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 
            marker='o',markerfacecolor='red', markersize=10)
    plt.title('accuracy vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))
'''
