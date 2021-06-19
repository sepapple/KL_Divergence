import acconeer.exptool as et
import csv
import os
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

    img1 = cv2.imread('/Users/sepa/Desktop/test/template/current_template_original.png',0)
    img2 = cv2.imread('/Users/sepa/Desktop/test/template/previous_template_original.png',0)
    # img1 = cv2.imread('/Users/sepa/Desktop/test/template/test.png',0)
    # img2 = cv2.imread('/Users/sepa/Desktop/test/template/test1.png',0)
    # img2 = cv2.imread('/Users/sepa/Desktop/test/template/test.png',0)
    # img2 = cv2.imread('/Users/sepa/Desktop/60GHzレーダーの実験/Euclidean_Distance/template/current_template.png',0)
    # print(img1)
    # exit(1)
 
    # ウィンドウサイズを設定
    # 1枚目の画像の(100,100)座標からwsize分の画像を抽出しテンプレート画像とする
    # wsize = 300
    # template = img1[100:100+wsize, 100:100+wsize]
    
    
    # テンプレートマッチング
    method = cv2.TM_CCOEFF_NORMED                               # Normalized Cross Correlation (NCC) 正規化相互相関係数
    
    res = cv2.matchTemplate(img1,img2,method)               # テンプレートマッチングの結果
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)     # 最小値と最大値、その位置を取得
    
    print(max_val, max_loc)


if __name__ == "__main__":
    main()
