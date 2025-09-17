import cv2
import numpy as np
import time

def compute_features(detector, img):
    def compute_features(detector, img):
        # detect keypoints in the image and compute descriptors
        kp = detector.detect(img, None)
        kp, des = detector.compute(img, kp)
        return kp, des

    def main():
        # create feature detectors and descriptors
        orb = cv2.ORB_create(nfeatures=1000)
        akaze = cv2.AKAZE_create()
        # tebld is in the xfeatures2d module (requires opencv-contrib-python)
        teblid = cv2.xfeatures2d.TEBLID_create(scale_factor=1.0, n_bits=cv2.xfeatures2d.TEBLID_SIZE_256_BITS)

        # load two images in grayscale
        img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            # print error if images cannot be loaded
            print("error loading images!")
            return

        # use orb to detect keypoints and compute descriptors; record the time taken
        t0 = time.time()
        kp1_orb, des1_orb = compute_features(orb, img1)
        kp2_orb, des2_orb = compute_features(orb, img2)
        orb_time = (time.time() - t0) * 1000

        # use orb for keypoint detection and teblid to compute descriptors; record the time taken
        t0 = time.time()
        kp1_teb = orb.detect(img1, None)
        kp2_teb = orb.detect(img2, None)
        _, des1_teb = teblid.compute(img1, kp1_teb)
        _, des2_teb = teblid.compute(img2, kp2_teb)
        teb_time = (time.time() - t0) * 1000

        # use akaze for detection and description; record the time taken
        t0 = time.time()
        kp1_akaze, des1_akaze = compute_features(akaze, img1)
        kp2_akaze, des2_akaze = compute_features(akaze, img2)
        akaze_time = (time.time() - t0) * 1000

        # match descriptors between images using bfmatcher with hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches_orb = bf.match(des1_orb, des2_orb)
        matches_teb = bf.match(des1_teb, des2_teb)
        matches_akaze = bf.match(des1_akaze, des2_akaze)

        # print orb results
        print("orb:")
        print(f"  keypoints: {len(kp1_orb)} in image1, {len(kp2_orb)} in image2")
        print(f"  time: {orb_time:.2f} ms, matches: {len(matches_orb)}")

        # print orb with teblid results
        print("orb with teblid:")
        print(f"  keypoints: {len(kp1_teb)} in image1, {len(kp2_teb)} in image2")
        print(f"  time: {teb_time:.2f} ms, matches: {len(matches_teb)}")

        # print akaze results
        print("akaze:")
        print(f"  keypoints: {len(kp1_akaze)} in image1, {len(kp2_akaze)} in image2")
        print(f"  time: {akaze_time:.2f} ms, matches: {len(matches_akaze)}")

        # draw matches using orb and teblid keypoints and display the result
        match_img = cv2.drawMatches(img1, kp1_teb, img2, kp2_teb, matches_teb, None, flags=2)
        cv2.imshow("orb with teblid matches", match_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if __name__ == '__main__':
        main()
