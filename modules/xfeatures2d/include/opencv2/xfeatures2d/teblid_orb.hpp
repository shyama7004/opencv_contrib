#ifndef OPENCV_XFEATURES2D_TEBLID_ORB_HPP
#define OPENCV_XFEATURES2D_TEBLID_ORB_HPP

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>

namespace cv
{
    namespace xfeatures2d
    {
        class CV_EXPORTS ORBwithTEBLID : public Feature2D
        {
        public:
            static const int SIZE_256_BITS = TEBLID::SIZE_256_BITS;
            static const int SIZE_512_BITS = TEBLID::SIZE_512_BITS;

            static Ptr<ORBwithTEBLID> create(int orb_nfeatures = 500,
                                            float orb_scale_factor = 1.2f,
                                            int orb_nlevels = 8,
                                            float teblid_scale_factor = 6.25f,
                                            int teblid_nbits = SIZE_256_BITS);

            void detect(InputArray image, std::vector<KeyPoint> &keypoints, InputArray mask = noArray()) override;
            void compute(InputArray image, std::vector<KeyPoint> &keypoints, OutputArray descriptors) override;
            void detectAndCompute(InputArray image, InputArray mask, std::vector<KeyPoint> &keypoints,
                                  OutputArray descriptors, bool useProvidedKeypoints = false) override;
            int descriptorSize() const override;
            int descriptorType() const override;
            int defaultNorm() const override;
            String getDefaultName() const override { return "ORBwithTEBLID"; }

            ORBwithTEBLID(int orb_nfeatures, float orb_scale_factor, int orb_nlevels,
                         float teblid_scale_factor, int teblid_nbits);

        private:
            Ptr<Feature2D> orb_;
            Ptr<TEBLID> teblid_;
        };
    }
}

#endif
