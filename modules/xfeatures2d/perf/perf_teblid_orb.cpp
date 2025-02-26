// This file is part of the OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
#include <opencv2/xfeatures2d/teblid_orb.hpp>

namespace opencv_test { namespace {

typedef perf::TestBaseWithParam<std::string> TeblidPerfTest;

#define TEBLID_TEST_IMAGES \
    "cv/detectors_descriptors_evaluation/images_datasets/leuven/img1.png", \
    "stitching/a3.png"

#ifdef OPENCV_ENABLE_NONFREE
PERF_TEST_P(TeblidPerfTest, compute, testing::Values(TEBLID_TEST_IMAGES))
{
    const std::string filename = getDataPath(GetParam());
    Mat img = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty()) << "Unable to load source image " << filename;

    // Use SURF for keypoint detection to generate a stable set of keypoints.
    Ptr<SURF> surf = SURF::create(400);
    vector<KeyPoint> keypoints;
    surf->detect(img, keypoints);
    CV_Assert(!keypoints.empty());

    // Create a TEBLID instance (using 256 bits in this test).
    float teblidScale = 5.0f;
    Ptr<xfeatures2d::TEBLID> teblid = xfeatures2d::TEBLID::create(teblidScale, xfeatures2d::TEBLID::SIZE_256_BITS);
    Mat descriptors;

    declare.in(img).time(90);
    TEST_CYCLE()
    {
        teblid->compute(img, keypoints, descriptors);
    }
    SANITY_CHECK_NOTHING();
}
#endif

}}
