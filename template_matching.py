import acconeer.exptool as et
import csv
import cv2
import os
import itertools

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

def main():
    #特定のセンサデータをプロット
    dir_name = '/Users/sepa/Desktop/template/experiment/'

    #グラフ全体を切り抜く設定
    beneath_height = 55
    top_height = 70
    left_width = 80
    right_width = 65

    #グラフ全体を切り抜く設定
    feature_beneath_height = 55
    feature_top_height = 70
    feature_left_width = 80
    feature_right_width = 65

    #特徴量全体を切り抜く設定
    # feature_beneath_height = 55
    # feature_top_height = 70
    # feature_left_width = 210
    # feature_right_width = 100

    #商品の特徴量を切り抜く設定
    # feature_beneath_height = 55
    # feature_top_height = 70
    # feature_left_width = 375
    # feature_right_width = 100

    #csv出力のための準備
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    header = []
    ccoeff = []
    ccoeff_normed = []
    ccorr = []
    ccorr_normed = []
    sqdiff = []
    sqdiff_normed = []

    header.append("method")
    ccoeff.append(methods[0])
    ccoeff_normed.append(methods[1])
    ccorr.append(methods[2]) 
    ccorr_normed.append(methods[3]) 
    sqdiff.append(methods[4]) 
    sqdiff_normed.append(methods[5])
    
    
    targets = [target_dir for target_dir in os.listdir(dir_name) if(not target_dir.startswith('.') and not 'csv' in target_dir and not 'result' in target_dir)]
    dir_paths = [dir_name+target+'/' for target in targets]
    file_paths = []
    for dir_path in dir_paths:
        temp=[]
        for filename in os.listdir(dir_path):
            if (not filename.startswith('.')):
                temp.append(dir_path+filename)
        file_paths.append(temp)
    
    '''
    #同じ条件でテンプレートマッチング(file_pathsの番号を変えることで入力データを変更可能 0:empty,1:bok1,2:book2)
    for comb in itertools.combinations(file_paths[2],2):
        target1 = comb[0].split('/')
        target2 = comb[1].split('/')
        print(str(target1[-2])+'/'+str(target1[-1])+' : '+str(target2[-2])+'/'+str(target2[-1]))
        header.append(str(target1[-2])+'/'+str(target1[-1])+' : '+str(target2[-2])+'/'+str(target2[-1]))
        img1 = cv2.imread(comb[0],0)
        img2 = cv2.imread(comb[1],0)
        # cv2.imshow('image',img1[top_height:-beneath_height,left_width:-right_width])
        # cv2.imshow('image',img1[top_height:-beneath_height,left_width:-right_width])

        for meth in methods:
            method = eval(meth)
            res = cv2.matchTemplate(img1[top_height:-beneath_height,left_width:-right_width],img2[feature_top_height:-feature_beneath_height,feature_left_width:-feature_right_width],method)               
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) 
            if method in [cv2.cv2.TM_CCOEFF]:
                ccoeff.append(max_val)
                print(str(meth)+': '+str(min_val))
            elif method in [cv2.cv2.TM_CCOEFF_NORMED]:
                ccoeff_normed.append(max_val)
                print(str(meth)+': '+str(max_val))
                # print(str(meth)+': '+str(max_loc))
            elif method in [cv2.TM_CCORR]:
                ccorr.append(max_val) 
                print(str(meth)+': '+str(max_val))
            elif method in [cv2.cv2.TM_CCORR_NORMED]:
                ccorr_normed.append(max_val) 
                print(str(meth)+': '+str(max_val))
            elif method in [cv2.cv2.TM_SQDIFF]:
                sqdiff.append(max_val) 
                print(str(meth)+': '+str(max_val))
            elif method in [cv2.TM_SQDIFF_NORMED]:
                sqdiff_normed.append(max_val)
                print(str(meth)+': '+str(max_val))

    df = pd.DataFrame([ccoeff,ccoeff_normed,ccorr,ccorr_normed,sqdiff,sqdiff_normed],columns=header)
    # df.to_csv("/Users/sepa/Desktop/template/experiment/result/graph_extract/empty_template_matching.csv",index=False)
    # df.to_csv("/Users/sepa/Desktop/template/experiment/result/graph_extract/book1_template_matching.csv",index=False)
    df.to_csv("/Users/sepa/Desktop/template/experiment/result/graph_extract/book2_template_matching.csv",index=False)
    exit(1)
    '''

    header = []
    ccoeff = []
    ccoeff_normed = []
    ccorr = []
    ccorr_normed = []
    sqdiff = []
    sqdiff_normed = []

    header.append("method")
    ccoeff.append(methods[0])
    ccoeff_normed.append(methods[1])
    ccorr.append(methods[2]) 
    ccorr_normed.append(methods[3]) 
    sqdiff.append(methods[4]) 
    sqdiff_normed.append(methods[5])

    #違う条件でテンプレートマッチング
    for i in file_paths[1]:
        for j in file_paths[2]:
            target1 = i.split('/')
            target2 = j.split('/')
            print(str(target1[-2])+'/'+str(target1[-1])+' : '+str(target2[-2])+'/'+str(target2[-1]))
            header.append(str(target1[-2])+'/'+str(target1[-1])+' : '+str(target2[-2])+'/'+str(target2[-1]))
            img1 = cv2.imread(i,0)
            img2 = cv2.imread(j,0)

            # cv2.imwrite(filename_1+'_Allextraction.png',img1[top_height:-beneath_height,left_width:-right_width])
            for meth in methods:
                method = eval(meth)
                res = cv2.matchTemplate(img1[top_height:-beneath_height,left_width:-right_width],img2[feature_top_height:-feature_beneath_height,feature_left_width:-feature_right_width],method)               
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) 
                if method in [cv2.cv2.TM_CCOEFF]:
                    ccoeff.append(max_val)
                    print(str(meth)+': '+str(min_val))
                elif method in [cv2.cv2.TM_CCOEFF_NORMED]:
                    ccoeff_normed.append(max_val)
                    print(str(meth)+': '+str(max_val))
                    # print(str(meth)+': '+str(max_loc))
                elif method in [cv2.TM_CCORR]:
                    ccorr.append(max_val) 
                    print(str(meth)+': '+str(max_val))
                elif method in [cv2.cv2.TM_CCORR_NORMED]:
                    ccorr_normed.append(max_val) 
                    print(str(meth)+': '+str(max_val))
                elif method in [cv2.cv2.TM_SQDIFF]:
                    sqdiff.append(max_val) 
                    print(str(meth)+': '+str(max_val))
                elif method in [cv2.TM_SQDIFF_NORMED]:
                    sqdiff_normed.append(max_val)
                    print(str(meth)+': '+str(max_val))

    df = pd.DataFrame([ccoeff,ccoeff_normed,ccorr,ccorr_normed,sqdiff,sqdiff_normed],columns=header)
    # df.to_csv("/Users/sepa/Desktop/template/experiment/result/graph_extract/empty_book1_template_matching.csv",index=False)
    # df.to_csv("/Users/sepa/Desktop/template/experiment/result/graph_extract/empty_book2_template_matching.csv",index=False)
    df.to_csv("/Users/sepa/Desktop/template/experiment/result/graph_extract/book1_book2_template_matching.csv",index=False)

    #保存
    # cv2.imwrite(filename_1+'_Allextraction.png',img1[top_height:-beneath_height,left_width:-right_width])
    # cv2.imwrite(filename_2+'_Allextraction.png',img2[top_height:-beneath_height,left_width:-right_width])
    # cv2.imwrite(filename_1+'_Featureextraction.png',img1[top_height:-beneath_height,left_width:-right_width])
    # cv2.imwrite(filename_2+'_Featureextraction.png',img2[top_height:-beneath_height,left_width:-right_width])


if __name__ == "__main__":
    main()
