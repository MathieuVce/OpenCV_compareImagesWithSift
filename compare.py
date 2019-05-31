import cv2 as cv
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim as ssim


img1 = cv.imread('./images/original_golden_bridge.jpg',0)
img2 = cv.imread('./images/sunburst.jpg',0)
# img2 = cv.imread('./images/textured.jpg',0)
# img2 = cv.imread('./images/blu_filer.jpg',0)
# img2 = cv.imread('./images/duplicate.jpg',0)



# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
if len(kp1) <= len(kp2):
        number_keypoints = len(kp1)
else:
        number_keypoints = len(kp2)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)


ratio = 0.7

good_points = []
matchesMask = [[0,0]
for i in range(len(matches))]
for i,(m,n) in enumerate(matches):
    if m.distance < ratio * n.distance:
        matchesMask[i]=[1,0]
        good_points.append(m)

if img1.shape == img2.shape:
        print("The images have same size and channels")
        s = ssim(img1, img2, multichannel = True)
        print("SSIM: %.2f" % s)
        print("Keypoints 1ST Image: " + str(len(kp1)))
        print("Keypoints 2ND Image: " + str(len(kp2)))
        print('matches : ' + str(len(good_points)))
        print("How good is the match: %.2f" % (len(good_points) / number_keypoints * 100))


        draw_params = dict(matchColor = (0,255,0),
                        matchesMask = matchesMask,
                        singlePointColor = (255,0,0),
                        flags = 0)


        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
        plt.imshow(img3,),plt.show()
