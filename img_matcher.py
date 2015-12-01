# coding: utf-8

import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import time
import math


MIN_MATCH_COUNT = 10
FEATURE_DETECTOR = "SIFT"
CROSS_CHECK = False
RECT_SLICE = 3

def contrast(image, a):
  lut = [ np.uint8(255.0 / (1 + math.exp(-a * (i - 128.) / 255.))) for i in range(256)] 
  result_image = np.array( [ lut[value] for value in image.flat], dtype=np.uint8 )
  result_image = result_image.reshape(image.shape)
  return result_image

# 矩形に切り出す
def get_rects( width, height, min_w, min_h, max_w, max_h, size ):
    
    rects = []
    while size > 0:
        while True:
            x = random.randint(0,width)
            y = random.randint(0,height)

            w = random.randint(min_w,max_w)
            h = random.randint(min_h,max_h)

            if not 1.0 <= w / h <= 3.0:
                continue

            if x + w <= width and y + h <= height:
                rects.append((x,y,w,h))
                break
        size -= 1
    
    return rects

def get_rects_by_rule( width, height, spr_limit ):
    
    rects = []

    for spr_x in range(1,spr_limit+1):
        for spr_y in range(1,spr_limit+1):
            for i in range(spr_x):
                for j in range(spr_y):
                    x = int(float(i)/spr_x*width)
                    y = int(float(j)/spr_y*height)

                    w = width / spr_x
                    h = height / spr_y

                    rects.append((x,y,w,h))
                    
                    x = int(float(i)/spr_x*width) + width/spr_x/2
                    y = int(float(j)/spr_y*height)

                    if x+w <= width and y+h <= height:
                        rects.append((x,y,w,h))

                    x = int(float(i)/spr_x*width)
                    y = int(float(j)/spr_y*height) + height/spr_y/2

                    if x+w <= width and y+h <= height:
                        rects.append((x,y,w,h))

                    x = int(float(i)/spr_x*width) + width/spr_x/2
                    y = int(float(j)/spr_y*height) + height/spr_y/2

                    if x+w <= width and y+h <= height:
                        rects.append((x,y,w,h))

    return rects


def cut_kp( kp, des, num ):
    a = sorted([ (kp[i],des[i]) for i in range(len(kp)) ], key=lambda (a,b): a.response, reverse=True )
    
    # 上位だけを残す
    # a = a[:num]

    # 順位で重みをつけて削る
    while len(a) > num:
        r = random.randint(0,len(a)*(1+len(a))/2)
        n = int( ( math.sqrt(1.0+8*r) - 1 ) / 2.0 )
        if n < 0:
            n = 0
        if n > len(a)-1:
            n = len(a)-1

        del a[n]

    # 一様確率で削る
#     while len(a) > num:
#         n = random.randint(0,len(a)-1)
#         del a[n]

    # random.shuffle(a)
    b =  [ a[i][0] for i in range(len(a)) ]
    c = np.array([ a[i][1] for i in range(len(a)) ])
    return b,c


# 矩形に含まれるkeypointを抽出
def get_keypoints( rect, kp, des ):
    x,y,w,h = rect
    kp_ext = []
    des_ext = []
    for i in range(len(kp)):
        if x <= kp[i].pt[0] <= x+w and y <= kp[i].pt[1] <= y+h:
            kp_ext.append(kp[i])
            des_ext.append(des[i])
    des_ext = np.array(des_ext)

    # kp_ext,des_ext = cut_kp(kp_ext,des_ext,100)

    return kp_ext,des_ext

# 面積
def shikakkei(A,B,C,D):

    # 交差していないと仮定
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    C = C.astype(np.float64)
    D = D.astype(np.float64)

    c = np.linalg.norm(A-C)
    a = np.linalg.norm(A-B)
    b = np.linalg.norm(B-C)
    s = (a + b + c)/2.0
    S1 = math.sqrt( s*(s-a)*(s-b)*(s-c) + 1.0e-10 )

    a = np.linalg.norm(C-D)
    b = np.linalg.norm(D-A)
    s = (a + b + c)/2.0
    S2 = math.sqrt( s*(s-a)*(s-b)*(s-c) + 1.0e-10 )

    return S1+S2

def kakudo_CAB(C,A,B):
    b = np.linalg.norm(A-C)
    c = np.linalg.norm(A-B)
    a = np.linalg.norm(B-C)
    
    cos_CAB = (b**2+c**2-a**2)/(2*b*c+1.0e-10)

    if cos_CAB > 1.0:
        cos_CAB = 1.0
    if cos_CAB < -1.0:
        cos_CAB = -1.0

    return math.acos(cos_CAB)


def kakudo(A,B,C,D):
    # 交差していないと仮定
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    C = C.astype(np.float64)
    D = D.astype(np.float64)

    return [kakudo_CAB(A,B,C),
            kakudo_CAB(B,C,D),
            kakudo_CAB(C,D,A),
            kakudo_CAB(D,A,B)]


# マッチング
def matching(kp1,des1,img1,kp2_ext,des2_ext):

    global MIN_MATCH_COUNT

    kp2 = kp2_ext
    des2 = des2_ext

    if len(kp1) <= MIN_MATCH_COUNT*2 or len(kp2) <= MIN_MATCH_COUNT*2:
        return [], None


    #FLANN_INDEX_KDTREE = 0
    #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    #search_params = dict(checks = 50)
    #flann = cv2.FlannBasedMatcher(index_params, search_params)
    #matches = flann.knnMatch(des1,des2,k=2)

    global FEATURE_DETECTOR
    global CROSS_CHECK

    if FEATURE_DETECTOR == "AKAZE":
        if CROSS_CHECK:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
            good = bf.match(des1,des2)
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.8*n.distance:
                    good.append(m)
    else:
        if CROSS_CHECK:
            bf = cv2.BFMatcher(crossCheck=True)
            good = bf.match(des1,des2)
        else:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.8*n.distance:
                    good.append(m)
 


    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is None:
            return [], None

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        return good, np.int32(dst)

    return [], None


# メイン処理
def multi_query_matching( query_img_filename, target_img_filename, result_save_filename ):

    start_time = time.time()

    img1 = cv2.imread(query_img_filename,0)
    img2 = cv2.imread(target_img_filename,0)
    
    # resize

    img1_w_max = 256
    img1_h_max = 256
    img2_w_max = 512
    img2_h_max = 512


    img1_w = img1.shape[1]
    img1_h = img1.shape[0]
    if img1_w > img1_w_max:
        img1_h = int(float(img1_h)*img1_w_max/img1_w)
        img1_w = img1_w_max
    if img1_h > img1_h_max:
        img1_w = int(float(img1_w)*img1_h_max/img1_h)
        img1_h = img1_h_max
    img2_w = img2.shape[1]
    img2_h = img2.shape[0]
    if img2_w > img2_w_max:
        img2_h = int(float(img2_h)*img2_w_max/img2_w)
        img2_w = img2_w_max
    if img2_h > img2_h_max:
        img2_w = int(float(img2_w)*img2_h_max/img2_h)
        img2_h = img2_h_max

    img1 = cv2.resize(img1, ( img1_w, img1_h ))
    img2 = cv2.resize(img2, ( img2_w, img2_h ))

    # Initiate detector
    if FEATURE_DETECTOR == "AKAZE":
        detector = cv2.AKAZE_create(nOctaveLayers=1)
    else:
        detector = cv2.xfeatures2d.SIFT_create(nfeatures=10000, sigma=1.0, contrastThreshold=0.01, edgeThreshold=20 )
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = detector.detectAndCompute(img1,None)
    kp2, des2 = detector.detectAndCompute(img2,None)
    
    kp1, des1 = cut_kp( kp1, des1, 200 )
    kp2, des2 = cut_kp( kp2, des2, 4000 )
    
    print "KP1: " + str(len(kp1))
    print "KP2: " + str(len(kp2))
    
    
    print time.time() - start_time
    
    rects = get_rects_by_rule( img2.shape[1], img2.shape[0], RECT_SLICE )
    
    print time.time() - start_time
    
    dsts = []
    for rect in rects:
        #x,y,w,h = rect
        #img2 = cv2.polylines(img2,[np.int32([[x,y],[x,y+h],[x+w,y+h],[x+w,y]])],True,255,3, cv2.LINE_AA)
        kp2_ext, des2_ext = get_keypoints( rect, kp2, des2 )
        good, dst = matching(kp1,des1,img1,kp2_ext,des2_ext)
        if not dst is None:
            dsts.append([0,dst])
    
    print time.time() - start_time

    # 面積が異常に小さいときは除外
    i = 0
    while i < len(dsts):
        if shikakkei(dsts[i][1][0],dsts[i][1][1],dsts[i][1][2],dsts[i][1][3]) < 2.0:
           del dsts[i]
        else:
            i += 1
 
    
    for i in range(len(dsts)):
        k = kakudo(dsts[i][1][0],dsts[i][1][1],dsts[i][1][2],dsts[i][1][3])
        k.sort()
        dsts[i][0] = k[0]
    
    
    dsts = sorted( dsts, key=lambda x:x[0],reverse=True)
    
    i = 0
    while i < len(dsts):
        if dsts[i][0] < 1:
           del dsts[i]
        else:
            i += 1
    
    # クラスタリング
    for i in range(len(dsts)):
        dsts[i].append(-1)
    for i in range(len(dsts)):
        if dsts[i][2] < 0:
            dsts[i][2] = i
            for j in range(i,len(dsts)):
                dist =(  np.linalg.norm(dsts[i][1][0]-dsts[j][1][0])
                       + np.linalg.norm(dsts[i][1][1]-dsts[j][1][1])
                       + np.linalg.norm(dsts[i][1][2]-dsts[j][1][2])
                       + np.linalg.norm(dsts[i][1][3]-dsts[j][1][3]) ) / (
                         np.linalg.norm(dsts[i][1][0]-dsts[i][1][1]) 
                      + np.linalg.norm(dsts[i][1][1]-dsts[i][1][2]) 
                       + np.linalg.norm(dsts[i][1][2]-dsts[i][1][3]) 
                       + np.linalg.norm(dsts[i][1][3]-dsts[i][1][0])
                      +1.0e-10)
                if dist < 1.0 and dsts[j][2] < 0:
                    dsts[j][2] = i
    
    dsts_n = []
    for i in range(len(dsts)):
        dsts_n_i = []
        for j in range(len(dsts)):
            if dsts[j][2] == i:
                dsts_n_i.append(dsts[j])
        # 平均をとる
        if len(dsts_n_i) > 0:
            a = 0
            b = 0
            c = 0
            d = 0
            for j in range(len(dsts_n_i)):
                a += dsts_n_i[j][1][0]
                b += dsts_n_i[j][1][1]
                c += dsts_n_i[j][1][2]
                d += dsts_n_i[j][1][3]
            a /= len(dsts_n_i)
            b /= len(dsts_n_i)
            c /= len(dsts_n_i)
            d /= len(dsts_n_i)
            dsts_n.append(dsts_n_i[0])
            dsts_n[-1][1][0] = a
            dsts_n[-1][1][1] = b
            dsts_n[-1][1][2] = c
            dsts_n[-1][1][3] = d
    
    
    for i in range(len(dsts_n)):
        img2 = cv2.polylines(img2,[dsts_n[i][1]],True,255,3, cv2.LINE_AA)


    if result_save_filename is None:
        out = cv2.drawKeypoints(img1, kp1, None)
        plt.imshow(out,'gray'),plt.show()
        
        out = cv2.drawKeypoints(img2, kp2, None)
        plt.imshow(out,'gray'),plt.show()
        
        plt.imshow(img2, 'gray'),plt.show()
    else:
        out = cv2.drawKeypoints(img1, kp1, None)
        cv2.imwrite(result_save_filename+".1.jpg",out)
        out = cv2.drawKeypoints(img2, kp2, None)
        cv2.imwrite(result_save_filename+".2.jpg",out)

        cv2.imwrite(result_save_filename,img2)

    # クエリーの座標を返したい
    # もとの縮尺の座標と、縮尺した画像の座標
    return []
    
   
    
