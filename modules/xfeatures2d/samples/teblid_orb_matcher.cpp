#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/teblid_orb.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
using namespace std::chrono;

struct PerformanceMetrics {
    double detection_time;
    double matching_time;
    int num_keypoints1;
    int num_keypoints2;
    int num_matches;
    int num_good_matches;
};

/**
 * @brief Detect and match keypoints between two images using ORBwithTEBLID.
 * Directly concatenates the binary descriptors from ORB and TEBLID.
 */
PerformanceMetrics detectAndMatch(const Mat& img1, const Mat& img2, Mat& output_img) {
    PerformanceMetrics metrics = {};

    auto t_start = high_resolution_clock::now();

    Ptr<ORBwithTEBLID> detector = ORBwithTEBLID::create(
        3000,        // orb_nfeatures
        1.2f,        // orb_scaleFactor
        8,           // orb_nlevels
        1.0f,        // teblid_scale_factor
        ORBwithTEBLID::SIZE_512_BITS,
        31,          // edgeThreshold
        0,           // firstLevel
        4,           // WTA_K
        ORB::HARRIS_SCORE,
        31,          // patchSize
        10           // fastThreshold
    );

    if (detector.empty()) {
        throw runtime_error("Failed to create ORBwithTEBLID detector.");
    }

    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1, false);
    detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2, false);

    auto t_detection_end = high_resolution_clock::now();
    metrics.detection_time = duration_cast<milliseconds>(t_detection_end - t_start).count();
    metrics.num_keypoints1 = static_cast<int>(keypoints1.size());
    metrics.num_keypoints2 = static_cast<int>(keypoints2.size());

    if (descriptors1.empty() || descriptors2.empty()) {
        throw runtime_error("Failed to compute descriptors or no keypoints found.");
    }

    BFMatcher matcher(NORM_HAMMING, false);
    vector<vector<DMatch>> knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

    auto t_matching_end = high_resolution_clock::now();
    metrics.matching_time = duration_cast<milliseconds>(t_matching_end - t_detection_end).count();

    const float ratio_thresh = 0.8f;
    vector<DMatch> good_matches;
    for (auto &m : knn_matches) {
        if (m.size() < 2) continue;
        float ratio = m[0].distance / m[1].distance;
        if (ratio < ratio_thresh) {
            good_matches.push_back(m[0]);
        }
    }

    metrics.num_matches = static_cast<int>(knn_matches.size());
    metrics.num_good_matches = static_cast<int>(good_matches.size());

    drawMatches(img1, keypoints1, img2, keypoints2, good_matches, output_img,
                Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    return metrics;
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            cerr << "Usage: " << argv[0] << " <img1> <img2>\n";
            return -1;
        }

        Mat img1 = imread(argv[1], IMREAD_GRAYSCALE);
        Mat img2 = imread(argv[2], IMREAD_GRAYSCALE);

        if (img1.empty() || img2.empty()) {
            throw runtime_error("Could not load input images!");
        }

        Mat output_img;
        PerformanceMetrics metrics = detectAndMatch(img1, img2, output_img);

        cout << "\n[ORBwithTEBLID Metrics]\n";
        cout << " * Detection time   : " << metrics.detection_time << " ms\n";
        cout << " * Matching time    : " << metrics.matching_time << " ms\n";
        cout << " * Keypoints image1 : " << metrics.num_keypoints1 << "\n";
        cout << " * Keypoints image2 : " << metrics.num_keypoints2 << "\n";
        cout << " * KNN matches      : " << metrics.num_matches << "\n";
        cout << " * Good matches     : " << metrics.num_good_matches << "\n";

        namedWindow("ORBwithTEBLID Matches", WINDOW_NORMAL);
        imshow("ORBwithTEBLID Matches", output_img);
        imwrite("ORBwithTEBLID_Matches.jpg", output_img);

        cout << "Press any key to exit." << endl;
        waitKey(0);
        return 0;

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return -1;
    }
}
