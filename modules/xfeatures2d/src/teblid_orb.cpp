#include "opencv2/xfeatures2d/teblid_orb.hpp"
#include "precomp.hpp"
#include <iostream>

namespace cv
{
    namespace xfeatures2d
    {

        ORBwithTEBLID::ORBwithTEBLID(
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
            int fastThreshold)
        {
            // Create ORB with enhanced settings (e.g., WTA_K=4 for better performance)
            orb_ = ORB::create(
                orb_nfeatures,
                orb_scaleFactor,
                orb_nlevels,
                edgeThreshold,
                firstLevel,
                WTA_K,
                scoreType,
                patchSize,
                fastThreshold);
            CV_Assert(!orb_.empty() && "[ORBwithTEBLID] Failed to create ORB instance.");

            // Create TEBLID (default to 512 bits for stronger matching, but can be set to 256 bits)
            teblid_ = TEBLID::create(teblid_scale_factor, teblid_nbits);
            CV_Assert(!teblid_.empty() && "[ORBwithTEBLID] Failed to create TEBLID instance.");
        }

        Ptr<ORBwithTEBLID> ORBwithTEBLID::create(
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
            int fastThreshold)
        {
            return makePtr<ORBwithTEBLID>(
                orb_nfeatures,
                orb_scaleFactor,
                orb_nlevels,
                teblid_scale_factor,
                teblid_nbits,
                edgeThreshold,
                firstLevel,
                WTA_K,
                scoreType,
                patchSize,
                fastThreshold);
        }

        void ORBwithTEBLID::detect(InputArray image, std::vector<cv::KeyPoint> &keypoints, InputArray mask)
        {
            // Use ORB detector for keypoint detection
            orb_->detect(image, keypoints, mask);
        }

        void ORBwithTEBLID::compute(InputArray image, std::vector<cv::KeyPoint> &keypoints, OutputArray descriptors)
        {
            if (keypoints.empty())
            {
                descriptors.release();
                return;
            }

            // Compute descriptors separately using ORB and TEBLID
            Mat orbDesc, teblidDesc;
            orb_->compute(image, keypoints, orbDesc);
            teblid_->compute(image, keypoints, teblidDesc);

            if (orbDesc.empty() || teblidDesc.empty())
            {
                descriptors.release();
                return;
            }
            CV_Assert(orbDesc.rows == teblidDesc.rows && "[ORBwithTEBLID::compute] Mismatch in descriptor counts.");

            // Convert descriptors to float for weighted combination
            Mat orbDesc_f, teblidDesc_f;
            orbDesc.convertTo(orbDesc_f, CV_32F);
            teblidDesc.convertTo(teblidDesc_f, CV_32F);

            // Apply weights (tunable parameters)
            const float orb_weight = 0.7f;
            const float teblid_weight = 0.3f;
            orbDesc_f *= orb_weight;
            teblidDesc_f *= teblid_weight;

            // Concatenate descriptors horizontally
            Mat concatenated_desc;
            hconcat(orbDesc_f, teblidDesc_f, concatenated_desc);

            // Normalize combined descriptor to improve matching robustness
            Mat normalized_desc;
            normalize(concatenated_desc, normalized_desc, 0, 1, NORM_MINMAX, -1, Mat());

            // Convert normalized descriptor back to binary (CV_8U)
            Mat final_desc;
            normalized_desc.convertTo(final_desc, CV_8U);

            final_desc.copyTo(descriptors);
        }

        void ORBwithTEBLID::detectAndCompute(InputArray image, InputArray mask,
                                             std::vector<cv::KeyPoint> &keypoints,
                                             OutputArray descriptors,
                                             bool useProvidedKeypoints)
        {
            if (!useProvidedKeypoints)
            {
                detect(image, keypoints, mask);
            }
            compute(image, keypoints, descriptors);
        }

        int ORBwithTEBLID::descriptorSize() const
        {
            // ORB typically provides 32 bytes; TEBLID provides 32 bytes when using 256 bits.
            return orb_->descriptorSize() + teblid_->descriptorSize();
        }

        int ORBwithTEBLID::descriptorType() const
        {
            // Both ORB and TEBLID produce binary descriptors.
            CV_Assert(orb_->descriptorType() == CV_8U && teblid_->descriptorType() == CV_8U);
            return CV_8U;
        }

        int ORBwithTEBLID::defaultNorm() const
        {
            // Binary descriptors use Hamming distance for matching.
            return NORM_HAMMING;
        }

    } // namespace xfeatures2d
} // namespace cv
