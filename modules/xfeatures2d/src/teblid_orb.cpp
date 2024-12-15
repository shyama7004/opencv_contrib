#include "opencv2/xfeatures2d/teblid_orb.hpp"
#include "precomp.hpp"
#include <iostream>

using namespace std;

namespace cv
{
    namespace xfeatures2d
    {
        ORBwithTEBLID::ORBwithTEBLID(int orb_nfeatures, float orb_scale_factor, int orb_nlevels,
                                     float teblid_scale_factor, int teblid_nbits)
        {
            orb_ = ORB::create(orb_nfeatures, orb_scale_factor, orb_nlevels);
            CV_Assert(!orb_.empty() && "ORB is empty");
            cout << "Initialized ORB with " << orb_nfeatures << " features" << endl;
            cout << "Scale factor: " << orb_scale_factor << endl;
            cout << "Number of levels: " << orb_nlevels << endl;

            teblid_ = TEBLID::create(teblid_scale_factor, teblid_nbits);
            CV_Assert(!teblid_.empty() && "TEBLID is empty");
            cout << "TEBLID scale factor: " << teblid_scale_factor << endl;
            cout << "Number of bits: " << teblid_nbits << endl;
            cout << "**********************************************************************************************************************************************************************************************" << endl;
            cout << "ORB descriptor size in bytes: " << orb_->descriptorSize() << endl;
            cout << "TEBLID descriptor size in bytes: " << teblid_->descriptorSize() << endl;
        }

        Ptr<ORBwithTEBLID> ORBwithTEBLID::create(int orb_nfeatures, float orb_scale_factor, int orb_nlevels,
                                               float teblid_scale_factor, int teblid_nbits)
        {
            return makePtr<ORBwithTEBLID>(orb_nfeatures, orb_scale_factor, orb_nlevels,
                                         teblid_scale_factor, teblid_nbits);
        }

        void ORBwithTEBLID::detect(InputArray image, vector<KeyPoint> &keypoints, InputArray mask)
        {
            orb_->detect(image, keypoints, mask);
        }

        void ORBwithTEBLID::compute(InputArray image, vector<KeyPoint> &keypoints, OutputArray descriptors)
        {
            cv::Mat orb_descriptors, teblid_descriptors;
            orb_->compute(image, keypoints, orb_descriptors);
            cout << "ORB descriptors " << orb_descriptors.rows << "x" << orb_descriptors.cols << endl;
            teblid_->compute(image, keypoints, teblid_descriptors);
            cout << "TEBLID descriptors " << teblid_descriptors.rows << "x" << teblid_descriptors.cols << endl;

            if (orb_descriptors.empty() || teblid_descriptors.empty())
            {
                descriptors.release();
                cout << "Empty descriptors" << endl;
                return;
            }

            CV_Assert(orb_descriptors.rows == teblid_descriptors.rows && "There is a mismatch in the number of descriptors");
            cv::hconcat(orb_descriptors, teblid_descriptors, descriptors);
            cout << "Combined descriptors " << descriptors.rows() << "x" << descriptors.cols() << endl;
        }

        void ORBwithTEBLID::detectAndCompute(InputArray image, InputArray mask,
                                            vector<KeyPoint> &keypoints, OutputArray descriptors,
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
            int orb_size = orb_->descriptorSize();
            int teblid_size = teblid_->descriptorSize();
            int combined_size = orb_size + teblid_size;
            cout << "ORB descriptor size: " << orb_size << " bytes, TEBLID descriptor size: " << teblid_size << " bytes, Combined descriptor size: " << combined_size << " bytes" << endl;
            return combined_size;
        }

        int ORBwithTEBLID::descriptorType() const
        {
            CV_Assert(orb_->descriptorType() == CV_8U && teblid_->descriptorType() == CV_8U);
            return CV_8U;
        }

        int ORBwithTEBLID::defaultNorm() const
        {
            return NORM_HAMMING;
        }
    } // namespace xfeatures2d
} // namespace cv
