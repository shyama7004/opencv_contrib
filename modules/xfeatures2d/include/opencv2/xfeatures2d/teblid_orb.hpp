#ifndef OPENCV_XFEATURES2D_TEBLID_ORB_HPP
#define OPENCV_XFEATURES2D_TEBLID_ORB_HPP

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

/**
 * @brief ORBwithTEBLID combines ORB keypoint detection with the TEBLID descriptor.
 *
 * This class uses the fast ORB detector to extract keypoints and then computes descriptors
 * using TEBLID. The weighted combination of both descriptors produces a more discriminative binary descriptor.
 */
namespace cv
{
    namespace xfeatures2d
    {

        class CV_EXPORTS ORBwithTEBLID : public cv::Feature2D
        {
        public:
            // TEBLID descriptor bit sizes.
            static const int SIZE_256_BITS = TEBLID::SIZE_256_BITS;
            static const int SIZE_512_BITS = TEBLID::SIZE_512_BITS;

            /**
             * @brief Creates an ORBwithTEBLID instance with the given parameters.
             *
             * @param orb_nfeatures Number of keypoints to detect.
             * @param orb_scaleFactor Pyramid decimation ratio.
             * @param orb_nlevels Number of pyramid levels.
             * @param teblid_scale_factor Scale factor for TEBLID descriptor.
             * @param teblid_nbits Descriptor length (choose SIZE_256_BITS or SIZE_512_BITS).
             * @param edgeThreshold Size of the border where features are not detected.
             * @param firstLevel Level of pyramid to put source image.
             * @param WTA_K Number of points that produce each element of the oriented BRIEF descriptor.
             * @param scoreType Algorithm to rank features (HARRIS_SCORE or FAST_SCORE).
             * @param patchSize Size of the patch used by the oriented BRIEF descriptor.
             * @param fastThreshold Threshold for the FAST keypoint detector.
             * @return Ptr to the created ORBwithTEBLID instance.
             */
            static cv::Ptr<ORBwithTEBLID> create(
                int orb_nfeatures = 3000,
                float orb_scaleFactor = 1.2f,
                int orb_nlevels = 8,
                float teblid_scale_factor = 1.0f,
                int teblid_nbits = SIZE_512_BITS,
                int edgeThreshold = 31,
                int firstLevel = 0,
                int WTA_K = 4,
                cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE,
                int patchSize = 31,
                int fastThreshold = 10);

            // Feature2D interface
            virtual void detect(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints,
                                cv::InputArray mask = cv::noArray()) override;

            virtual void compute(cv::InputArray image,
                                 std::vector<cv::KeyPoint> &keypoints,
                                 cv::OutputArray descriptors) override;

            virtual void detectAndCompute(cv::InputArray image, cv::InputArray mask,
                                          std::vector<cv::KeyPoint> &keypoints,
                                          cv::OutputArray descriptors,
                                          bool useProvidedKeypoints = false) override;

            virtual int descriptorSize() const override;
            virtual int descriptorType() const override;
            virtual int defaultNorm() const override;

            virtual cv::String getDefaultName() const override { return "ORBwithTEBLID"; }

            /**
             * @brief Constructor. Use the create() function to instantiate.
             */
            ORBwithTEBLID(
                int orb_nfeatures,
                float orb_scaleFactor,
                int orb_nlevels,
                float teblid_scale_factor,
                int teblid_nbits,
                int edgeThreshold,
                int firstLevel,
                int WTA_K,
                cv::ORB::ScoreType scoreType,
                int patchSize,
                int fastThreshold);

        private:
            cv::Ptr<cv::Feature2D> orb_;              // ORB detector and descriptor extractor.
            cv::Ptr<cv::xfeatures2d::TEBLID> teblid_; // TEBLID descriptor extractor.
        };

    } // namespace xfeatures2d
} // namespace cv

#endif // OPENCV_XFEATURES2D_TEBLID_ORB_HPP
