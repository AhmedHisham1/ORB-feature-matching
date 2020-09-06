import numpy as np
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist


def FAST(img, N=9, threshold=0.15, nms_window=2):
    kernel = np.array([[1,2,1],
                       [2,4,2],
                       [1,2,1]])/16      # 3x3 Gaussian Window
    
    img = convolve2d(img, kernel, mode='same')

    cross_idx = np.array([[3,0,-3,0], [0,3,0,-3]])
    circle_idx = np.array([[3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1,0,1,2,3],
	                       [0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1]])

    corner_img = np.zeros(img.shape)
    keypoints = []
    for y in range(3, img.shape[0]-3):
        for x in range(3, img.shape[1]-3):
            Ip = img[y,x]
            t = threshold*Ip
            # fast checking cross idx only
            if np.count_nonzero(Ip+t < img[y+cross_idx[0,:], x+cross_idx[1,:]]) >= 3 or np.count_nonzero(Ip-t > img[y+cross_idx[0,:], x+cross_idx[1,:]]) >= 3:
                # detailed check -> full circle
                if np.count_nonzero(img[y+circle_idx[0,:], x+circle_idx[1,:]] >= Ip+t) >= N or np.count_nonzero(img[y+circle_idx[0,:], x+circle_idx[1,:]] <= Ip-t) >= N:
                    # Keypoint [corner]
                    keypoints.append([x,y])
                    corner_img[y,x] = np.sum(np.abs(Ip - img[y+circle_idx[0,:], x+circle_idx[1,:]]))
            
    # NMS - Non Maximal Suppression
    fewer_kps = []
    for [x, y] in keypoints:
        window = corner_img[y-nms_window:y+nms_window+1, x-nms_window:x+nms_window+1]
        # v_max = window.max()
        loc_y_x = np.unravel_index(window.argmax(), window.shape)
        x_new = x + loc_y_x[1] - nms_window
        y_new = y + loc_y_x[0] - nms_window
        new_kp = [x_new, y_new]
        if new_kp not in fewer_kps:
            fewer_kps.append(new_kp)

    return np.array(fewer_kps)

def corner_orientations(img, corners, r_size=3):
    # r_size must be odd to have one centre point which is the corner
    mrows2 = int((r_size - 1) / 2)
    mcols2 = int((r_size - 1) / 2)
    
    # Padding to avoid errors @ corners near image edges. 
    # Padding value=0 to not affect the orientation calculations
    img = np.pad(img, (mrows2, mcols2), mode='constant', constant_values=0)

    # Calculating orientation by the intensity centroid method
    orientations = []
    for corner in corners:
        c0, r0 = corner
        m01, m10 = 0, 0
        for r in range(r_size):
            m01_temp = 0
            for c in range(r_size):
                I = img[r0+r, c0+c]
                m10 = m10 + I*(c-mcols2)
                m01_temp = m01_temp + I
            m01 = m01 + m01_temp*(r-mrows2)
        orientations.append(np.arctan2(m01, m10))

    return np.array(orientations)


def BRIEF(img, keypoints, n=256, patch_size=31, sigma=1, sample_seed=1):
    '''
    BRIEF [Binary Robust Independent Elementary Features] keypoint/corner descriptor
    '''
    random = np.random.RandomState(seed=sample_seed)

    kernel = np.array([[1,2,1],
                       [2,4,2],
                       [1,2,1]])/16      # 3x3 Gaussian Window
    
    img = convolve2d(img, kernel, mode='same')
    samples = (patch_size / 5.0) * random.randn(n*8)
    samples = np.array(samples, dtype=np.int32)
    samples = samples[(samples < (patch_size // 2)) & (samples > - (patch_size - 2) // 2)]

    pos1 = samples[:n * 2].reshape(n, 2)
    pos2 = samples[n * 2:n * 4].reshape(n, 2)

    rows = img.shape[0]
    cols = img.shape[1]

    mask = (((patch_size//2 - 1) < keypoints[:, 0])
            & (keypoints[:, 0] < (rows - patch_size//2 + 1))
            & ((patch_size//2 - 1) < keypoints[:, 1])
            & (keypoints[:, 1] < (cols - patch_size//2 + 1)))

    keypoints = np.array(keypoints[mask, :], dtype=np.intp, order='C', copy=False)

    descriptors = np.zeros((keypoints.shape[0], n), dtype=bool, order='C')

    for p in range(pos1.shape[0]):
            pr0 = pos1[p, 0]
            pc0 = pos1[p, 1]
            pr1 = pos2[p, 0]
            pc1 = pos2[p, 1]
            for k in range(keypoints.shape[0]):
                kr = keypoints[k, 0]
                kc = keypoints[k, 1]
                if img[kr + pr0, kc + pc0] < img[kr + pr1, kc + pc1]:
                    descriptors[k, p] = True
    return descriptors


def match(descriptors1, descriptors2, max_distance=np.inf, cross_check=True, max_ratio=1):
    distances = cdist(descriptors1, descriptors2, metric='euclidean')   # distances.shape: [len(d1), len(d2)]
    indices1 = np.arange(descriptors1.shape[0])     # [0, 1, 2, 3, 4, 5, 6, 7, ...]
    indices2 = np.argmin(distances, axis=1)         # [12, 465, 23, 111, 123, 45, 67, 2, 265, ....]

    if cross_check:
        matches1 = np.argmin(distances, axis=0)     # sorted indices of d1 distances
        mask = indices1 == matches1[indices2]       # 
        indices1 = indices1[mask]
        indices2 = indices2[mask]
    
    if max_distance < np.inf:
        mask = distances[indices1, indices2] < max_distance
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if max_ratio < 1.0:
        best_distances = distances[indices1, indices2]
        distances[indices1, indices2] = np.inf
        second_best_indices2 = np.argmin(distances[indices1], axis=1)
        second_best_distances = distances[indices1, second_best_indices2]
        second_best_distances[second_best_distances == 0] = np.finfo(np.double).eps
        ratio = best_distances / second_best_distances
        mask = ratio < max_ratio
        indices1 = indices1[mask]
        indices2 = indices2[mask]
    
    matches = np.column_stack((indices1, indices2))
    return matches



if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    from time import time

    img = cv2.imread('images/chess.jpg')
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    
    print('corner orientations')
    t3 = time()
    orientations = corner_orientations(gray, keypoints, r_size=9)
    print('orientations time: ', time()-t3)
    print(np.rad2deg(orientations))

    plt.figure()
    plt.imshow(original_img)
    for i in range(keypoints.shape[0]):
        plt.quiver(keypoints[i, 0], keypoints[i, 1], np.cos(orientations[i]), np.sin(orientations[i]), 
                width=0.002, headwidth=5, scale=30)
    plt.show()

    d1 = BRIEF(gray, keypoints)


    img = cv2.imread('images/waffle.jpg')
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    
    print('corner orientations')
    t3 = time()
    orientations = corner_orientations(gray, keypoints, r_size=9)
    print('orientations time: ', time()-t3)
    print(np.rad2deg(orientations))

    plt.figure()
    plt.imshow(original_img)
    for i in range(keypoints.shape[0]):
        plt.quiver(keypoints[i, 0], keypoints[i, 1], np.cos(orientations[i]), np.sin(orientations[i]), 
                width=0.002, headwidth=5, scale=30)
    plt.show()

    d2 = BRIEF(gray, keypoints)
    match(d1, d2)