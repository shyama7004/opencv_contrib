#include "test_precomp.hpp"
#include "opencv2/xfeatures2d/teblid_orb.hpp"

namespace opencv_test
{
    namespace
    {

        using namespace std;
        using namespace cv;
        using namespace cv::xfeatures2d;

        /**
         * @brief Basic detection and description test for ORBwithTEBLID.
         * This test verifies that keypoints are detected, descriptors are computed,
         * and that the descriptor size is correct (32 ORB + 32 TEBLID for 256-bit configuration).
         */
        TEST(XFeatures2d_TeblidORB, BasicDetectionAndDescription)
        {
            Mat img = Mat::zeros(Size(800, 800), CV_8UC1);
            // Draw synthetic features: circle, rectangle, and lines
            circle(img, Point(400, 400), 200, Scalar::all(255), -1);
            rectangle(img, Point(300, 300), Point(500, 500), Scalar::all(0), -1);
            line(img, Point(400, 200), Point(400, 600), Scalar::all(255), 5);
            line(img, Point(200, 400), Point(600, 400), Scalar::all(255), 5);

            Ptr<ORBwithTEBLID> detector = ORBwithTEBLID::create(2000, 1.2f, 8, 1.0f, ORBwithTEBLID::SIZE_256_BITS);
            ASSERT_FALSE(detector.empty());

            vector<KeyPoint> keypoints;
            Mat descriptors;
            detector->detectAndCompute(img, noArray(), keypoints, descriptors);

            ASSERT_FALSE(keypoints.empty()) << "No keypoints detected.";
            ASSERT_FALSE(descriptors.empty()) << "No descriptors computed.";
            EXPECT_EQ(descriptors.rows, static_cast<int>(keypoints.size()));
            EXPECT_EQ(detector->descriptorType(), CV_8U);
            EXPECT_EQ(detector->descriptorSize(), 64) << "Descriptor size should be 64 bytes (32 ORB + 32 TEBLID for 256 bits).";

            // Check that each descriptor is not all zeros.
            for (int i = 0; i < descriptors.rows; ++i)
            {
                bool all_zero = true;
                for (int j = 0; j < descriptors.cols; ++j)
                {
                    if (descriptors.at<uchar>(i, j) != 0)
                    {
                        all_zero = false;
                        break;
                    }
                }
                EXPECT_FALSE(all_zero) << "Descriptor " << i << " is all zeros.";
            }
        }

        /**
         * @brief Compare matching quality between two images with rotation.
         * This test verifies that ORBwithTEBLID finds a sufficient number of good matches
         * when one image is a rotated version of the other.
         */
        TEST(XFeatures2d_TeblidORB, CompareInliers)
        {
            Mat img1 = Mat::zeros(Size(800, 800), CV_8UC1);
            // Draw synthetic features
            circle(img1, Point(400, 400), 200, Scalar::all(255), -1);
            rectangle(img1, Point(300, 300), Point(500, 500), Scalar::all(0), -1);
            line(img1, Point(400, 200), Point(400, 600), Scalar::all(255), 5);
            line(img1, Point(200, 400), Point(600, 400), Scalar::all(255), 5);

            // Create a rotated version of the image.
            Mat img2;
            Point2f center(img1.cols / 2.0f, img1.rows / 2.0f);
            Mat rot = getRotationMatrix2D(center, 45, 1.0);
            warpAffine(img1, img2, rot, img1.size());

            Ptr<ORBwithTEBLID> detector = ORBwithTEBLID::create(2000, 1.2f, 8, 1.0f, ORBwithTEBLID::SIZE_256_BITS);
            ASSERT_FALSE(detector.empty());

            vector<KeyPoint> keypoints1, keypoints2;
            Mat descriptors1, descriptors2;
            detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
            detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);
            ASSERT_FALSE(keypoints1.empty()) << "No keypoints detected in first image.";
            ASSERT_FALSE(keypoints2.empty()) << "No keypoints detected in second image.";
            ASSERT_FALSE(descriptors1.empty()) << "No descriptors computed for first image.";
            ASSERT_FALSE(descriptors2.empty()) << "No descriptors computed for second image.";

            BFMatcher matcher(NORM_HAMMING);
            vector<vector<DMatch>> knn_matches;
            matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

            // Apply ratio test to find good matches.
            const float ratio_thresh = 0.8f;
            vector<DMatch> good_matches;
            for (size_t i = 0; i < knn_matches.size(); i++)
            {
                if (knn_matches[i].size() >= 2)
                {
                    float ratio = knn_matches[i][0].distance / knn_matches[i][1].distance;
                    if (ratio < ratio_thresh)
                    {
                        good_matches.push_back(knn_matches[i][0]);
                    }
                }
            }
            EXPECT_GT(static_cast<int>(good_matches.size()), 10) << "Insufficient number of good matches found.";
        }

        /**
         * @brief Test for low-texture images.
         * Ensures that the detector gracefully handles images with few features.
         */
        TEST(XFeatures2d_TeblidORB, LowTextureImage)
        {
            Mat img = Mat::ones(Size(800, 800), CV_8UC1) * 127; // nearly uniform image
            Ptr<ORBwithTEBLID> detector = ORBwithTEBLID::create(500, 1.2f, 8, 1.0f, ORBwithTEBLID::SIZE_256_BITS);
            vector<KeyPoint> keypoints;
            Mat descriptors;
            detector->detectAndCompute(img, noArray(), keypoints, descriptors);
            // Expect few or no keypoints in a low-texture image.
            EXPECT_LE(static_cast<int>(keypoints.size()), 20);
        }

    }
} // namespace opencv_test
