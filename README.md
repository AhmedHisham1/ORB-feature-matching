# ORB-feature-matching
Python implementation of ORB feature matching algorithm from scratch. (not using openCV)

This is a python implementation of the ORB feature extraction/detection and matching without using OpenCV orb functions. It was done as an **exercise of my understanding of the algorithm**.

##### Code Structure
Inside the ```utils.py``` file I have created the following functions:
- **```FAST():```** FAST algorithm [***Features from Accelerated Segment Test***] for keypoints (corners) detection.

- **```corner_orientations():```** a function that computes the orientations of each keypoint based on the ***intensity centroid*** method.

- **```BRIEF():```** BRIEF algorithm [***Binary Robust Independent Elementary Features***] for keypoints (corners) detection. The function can also utilize the keypoints orientations to compute the descriptors accordingly (steered-BRIEF algorithm).

- **```match():```** Brute force matching of the BRIEF descriptors based on hamming distance, with the option to perform cross-check.


The actual ORB implementation is in the ```ORB Notebook.ipynb``` file where I use all the functions of ```utils.py```.

## Examples
##### Multi-Scale Keypoints Detection
![keypoints](/images/out-kps.png)

##### Scale Test Matching
![keypoints](/images/out-matches.png)

##### Blur Test Matching
![keypoints](/images/out-matches2.png)


#### TODO:
The performance of this implementation is noticably worse (some incorrect matches & overall very slow) than the OpenCV ORB implementation, further improvements could be made in the future.