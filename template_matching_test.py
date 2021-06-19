import acconeer.exptool as et
import csv
import cv2

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
    
    #ダンボールと商品のピーク値を検出
    dic_name = '/Users/sepa/Desktop/template/compate/'
    filename_1 = '/Users/sepa/Desktop/template/compare/' + 'empty'
    filename_2 = '/Users/sepa/Desktop/template/compare/' + 'book1'
    filename_3 = '/Users/sepa/Desktop/template/compare/' + 'book2'
    filenames = []
    filenames.append(filename_1)
    filenames.append(filename_2)
    filenames.append(filename_3)
    
    for filename in filenames:
        df = np.loadtxt(filename+'.csv')
        plt.plot(range(len(df)),df,color='b')  
        plt.savefig(filename)
        plt.cla()
    


    img1 = cv2.imread(filename_1+'.png',0)
    img2 = cv2.imread(filename_2+'.png',0)
    #グラフ全体を切り抜く設定
    # beneath_height = 55
    # top_height = 70
    # left_width = 80
    # right_width = 65

    #特徴量の部分を切り抜く設定
    beneath_height = 55
    top_height = 70
    left_width = 210
    right_width = 100

    #表示方法(ESCでウィンドウを閉じる)
    # cv2.imshow('image',img1[top_height:-beneath_height,left_width:-right_width])
    # cv2.waitKey(0)

    #テンプレートマッチング
    method = cv2.TM_CCOEFF_NORMED                               
    res = cv2.matchTemplate(img1[top_height:-beneath_height,left_width:-right_width],img2[top_height:-beneath_height,left_width:-right_width],method)               # テンプレートマッチングの結果
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)     # 最小値と最大値、その位置を取得
    print(max_val, max_loc)

    #保存
    # cv2.imwrite(filename_1+'_Allextraction.png',img1[top_height:-beneath_height,left_width:-right_width])
    # cv2.imwrite(filename_2+'_Allextraction.png',img2[top_height:-beneath_height,left_width:-right_width])
    cv2.imwrite(filename_1+'_Featureextraction.png',img1[top_height:-beneath_height,left_width:-right_width])
    cv2.imwrite(filename_2+'_Featureextraction.png',img2[top_height:-beneath_height,left_width:-right_width])


if __name__ == "__main__":
    main()
