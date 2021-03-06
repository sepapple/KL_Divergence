import acconeer.exptool as et
import csv
import cv2
import os

#データ取り扱い
import pandas as pd
import numpy as np
import datetime
import time
import datetime
import math
from scipy import signal

#描画
import matplotlib
import matplotlib.pyplot as plt

def Euclidean_Distance(x,y):
    return np.sqrt(np.sum((x-y)**2))

def Cosine_Similarity(x,y):
    return np.dot(x,y)/(np.sqrt(np.dot(x,x))* np.sqrt(np.dot(y,y)))


def main():
    
    #特定のセンサデータをプロット
    dir_name = '/Users/sepa/Desktop/template/experiment/empty/'
    # filename_1 = '/Users/sepa/Desktop/template/compare/' + 'empty'
    # filename_2 = '/Users/sepa/Desktop/template/compare/' + 'book1'
    # filename_3 = '/Users/sepa/Desktop/template/compare/' + 'book2'

    #グラフ全体を切り抜く設定
    beneath_height = 55
    top_height = 70
    left_width = 90
    right_width = 65


    #特徴量全体を切り抜く設定
    feature_beneath_height = 55
    feature_top_height = 70
    feature_left_width = 210
    feature_right_width = 100

    #商品の特徴量を切り抜く設定
    product_feature_beneath_height = 55
    product_feature_top_height = 70
    product_feature_left_width = 375
    product_feature_right_width = 100

    files = os.listdir(dir_name)
    files_paths = [dir_name + file for file in files if(not file.startswith('.') and not 'csv' in file and not 'result' in file)]

    #表示方法(ESCでウィンドウを閉じる)
    for file in files_paths:
        img1 = cv2.imread(file,0)
        cv2.imwrite(dir_name+'GraphExtraction.png',img1[top_height:-beneath_height,left_width:-right_width])
        cv2.imwrite(dir_name+'AllFeatureExtraction.png',img1[feature_top_height:-feature_beneath_height,feature_left_width:-feature_right_width])
        cv2.imwrite(dir_name+'ProductFeatureExtraction.png',img1[product_feature_top_height:-product_feature_beneath_height,product_feature_left_width:-product_feature_right_width])
        exit(1)
        cv2.imshow('image',img1[top_height:-beneath_height,left_width:-right_width])
        cv2.waitKey(0)
    
    exit(1)
    #テンプレートマッチング
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    for meth in methods:
        method = eval(meth)
        res = cv2.matchTemplate(img1[top_height:-beneath_height,left_width:-right_width],img2[top_height:-beneath_height,left_width:-right_width],method)               
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) 
        print(str(meth)+'を用いた類似度: '+str(max_val))

    #保存
    # cv2.imwrite(filename_1+'_Allextraction.png',img1[top_height:-beneath_height,left_width:-right_width])
    # cv2.imwrite(filename_2+'_Allextraction.png',img2[top_height:-beneath_height,left_width:-right_width])
    # cv2.imwrite(filename_1+'_Featureextraction.png',img1[top_height:-beneath_height,left_width:-right_width])
    # cv2.imwrite(filename_2+'_Featureextraction.png',img2[top_height:-beneath_height,left_width:-right_width])


if __name__ == "__main__":
    main()
