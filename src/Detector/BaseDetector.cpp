#include "BaseDetector.h"
#include "MotionDetector.h"
#include "FaceDetector.h"
#include "PedestrianDetector.h"
#include "SSDMobileNetDetector.h"
#include "YoloDetector.h"

#ifdef BUILD_YOLO_LIB
#include "YoloDarknetDetector.h"
#endif
///
/// \brief CreateDetector
/// \param detectorType
/// \param gray
/// \return
///
BaseDetector* CreateDetector(
        tracking::Detectors detectorType,
        const config_t& config,
        cv::UMat& gray
        )
{
    BaseDetector* detector = nullptr;

    switch (detectorType)
    {
    case tracking::Motion_VIBE:
        detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_VIBE, gray);
        break;

    case tracking::Motion_MOG:
        detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_MOG, gray);
        break;

    case tracking::Motion_GMG:
        detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_GMG, gray);
        break;

    case tracking::Motion_CNT:
        detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_CNT, gray);
        break;

    case tracking::Motion_SuBSENSE:
        detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_SuBSENSE, gray);
        break;

    case tracking::Motion_LOBSTER:
        detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_LOBSTER, gray);
        break;

    case tracking::Motion_MOG2:
        detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_MOG2, gray);
        break;

    case tracking::Face_HAAR:
        detector = new FaceDetector(gray);
        break;

    case tracking::Pedestrian_HOG:
    case tracking::Pedestrian_C4:
        detector = new PedestrianDetector(gray);
        break;

    case tracking::SSD_MobileNet:
        detector = new SSDMobileNetDetector(gray);
        break;

    case tracking::Yolo_OCV:
        detector = new YoloOCVDetector(gray);
        break;

	case tracking::Yolo_Darknet:
#ifdef BUILD_YOLO_LIB
        detector = new YoloDarknetDetector(gray);
#else
		std::cerr << "Darknet inference engine was not configured in CMake" << std::endl;
#endif
		break;

    default:
        break;
    }

    if (!detector->Init(config))
    {
        delete detector;
        detector = nullptr;
    }
    return detector;
}
