#include "test_precomp.hpp"
#include "opencv2/xfeatures2d/teblid_orb.hpp"
#include <vector>

namespace opencv_test { namespace {

TEST(XFeatures2d_TeblidORB, BasicDetectionAndDescription)
{
    cv::Mat img = cv::Mat::zeros(cv::Size(800, 800), CV_8UC1);
    cv::circle(img, cv::Point(400, 400), 200, cv::Scalar::all(255), -1);
    cv::rectangle(img, cv::Point(300, 300), cv::Point(500, 500), cv::Scalar::all(0), -1);
    cv::line(img, cv::Point(400, 200), cv::Point(400, 600), cv::Scalar::all(255), 5);
    cv::line(img, cv::Point(200, 400), cv::Point(600, 400), cv::Scalar::all(255), 5);

    cv::Ptr<cv::xfeatures2d::ORBwithTEBLID> detector = cv::xfeatures2d::ORBwithTEBLID::create(
        2000,
        1.2f,
        8,
        1.0f,
        cv::xfeatures2d::ORBwithTEBLID::SIZE_256_BITS
    );
    ASSERT_FALSE(detector.empty());

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

    ASSERT_FALSE(keypoints.empty()) << "No keypoints detected.";
    ASSERT_FALSE(descriptors.empty()) << "No descriptors computed.";
    EXPECT_EQ(descriptors.rows, static_cast<int>(keypoints.size()));
    EXPECT_EQ(detector->descriptorType(), CV_8U);
    EXPECT_EQ(detector->descriptorSize(), 64);

    for (int i = 0; i < descriptors.rows; ++i) {
        bool all_zero = true;
        for (int j = 0; j < descriptors.cols; ++j) {
            if (descriptors.at<uchar>(i, j) != 0) {
                all_zero = false;
                break;
            }
        }
        EXPECT_FALSE(all_zero) << "Descriptor " << i << " is all zeros.";
    }
}

TEST(XFeatures2d_TeblidORB, CompareInliers)
{
    cv::Mat img1 = cv::Mat::zeros(cv::Size(800, 800), CV_8UC1);
    cv::circle(img1, cv::Point(400, 400), 200, cv::Scalar::all(255), -1);
    cv::rectangle(img1, cv::Point(300, 300), cv::Point(500, 500), cv::Scalar::all(0), -1);
    cv::line(img1, cv::Point(400, 200), cv::Point(400, 600), cv::Scalar::all(255), 5);
    cv::line(img1, cv::Point(200, 400), cv::Point(600, 400), cv::Scalar::all(255), 5);

    cv::Mat img2;
    cv::Point2f center(img1.cols / 2.0f, img1.rows / 2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(center, 45, 1.0);
    cv::warpAffine(img1, img2, rot, img1.size());

    cv::Ptr<cv::xfeatures2d::ORBwithTEBLID> detector = cv::xfeatures2d::ORBwithTEBLID::create(
        2000,
        1.2f,
        8,
        1.0f,
        cv::xfeatures2d::ORBwithTEBLID::SIZE_256_BITS
    );
    ASSERT_FALSE(detector.empty());

    std::vector<cv::KeyPoint> keypoints1;
    cv::Mat descriptors1;
    detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    ASSERT_FALSE(keypoints1.empty()) << "No keypoints detected in first image.";
    ASSERT_FALSE(descriptors1.empty()) << "No descriptors computed for first image.";

    std::vector<cv::KeyPoint> keypoints2;
    cv::Mat descriptors2;
    detector->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);
    ASSERT_FALSE(keypoints2.empty()) << "No keypoints detected in second image.";
    ASSERT_FALSE(descriptors2.empty()) << "No descriptors computed for second image.";

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

    const float ratio_thresh = 0.8f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i].size() >= 2) {
            float ratio = knn_matches[i][0].distance / knn_matches[i][1].distance;
            if (ratio < ratio_thresh) {
                good_matches.push_back(knn_matches[i][0]);
            }
        }
    }

    EXPECT_GT(static_cast<int>(good_matches.size()), 10) << "Insufficient number of good matches found.";
}

}} // namespace opencv_test
