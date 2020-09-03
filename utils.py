import numpy as np
from scipy.signal import convolve2d

def FAST(img, N=9, threshold=0.15):
    cross_idx = np.array([[3,0,-3,0], [0,3,0,-3]])
    circle_idx = np.array([[3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1,0,1,2,3],
	                       [0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1]])

    kernel = np.array([[1,2,1],
                       [2,4,2],
                       [1,2,1]])/16      # 3x3 Gaussian Window
    
    img = convolve2d(img, kernel, mode='same')

    keypoints = []
    for y in range(3, img.shape[0]-3):
        for x in range(3, img.shape[1]-3):
            Ip = img[y,x]
            t = threshold*Ip
            # fast checking cross idx only
            if np.count_nonzero(Ip+t < img[y+cross_idx[0,:], x+cross_idx[1,:]]) >= 3 or np.count_nonzero(Ip-t > img[y+cross_idx[0,:], x+cross_idx[1,:]]) >= 3:
                # detailed check -> full circle
                if np.count_nonzero(Ip+t < img[y+circle_idx[0,:], x+circle_idx[1,:]]) >= N or np.count_nonzero(Ip-t > img[y+circle_idx[0,:], x+circle_idx[1,:]]) >= N:
                    # Keypoint [corner]
                    keypoints.append([x,y])

    return keypoints


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    from time import time

    img = cv2.imread('images/waffle.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    t1 = time()
    keypoints = FAST(gray, N=9, threshold=0.15)
    print('me: ', time()-t1)
    features_img = np.copy(img)

    for keypoint in keypoints:
        features_img = cv2.circle(features_img, tuple(keypoint), 3, (0,255,0), 1)
    # features_img[keypoints] = [0,255,0]

    # fig = plt.figure(figsize=(10,10))
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.subplot(1,3,2)
    plt.imshow(features_img)
    
    # using cv2
    t2 = time()
    fast_cv2 = cv2.FastFeatureDetector_create()
    kp = fast_cv2.detect(img, None)
    print('cv: ', time()-t2)

    img_cv = img
    img_cv = cv2.drawKeypoints(img, kp, img_cv, color=(0,255,0))

    plt.subplot(1,3,3)
    plt.imshow(img_cv)
    plt.show()

    print('my keypoints: ', len(keypoints), '\ncv keypoints: ', len(kp))
    