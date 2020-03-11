#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <map>

#include <opencv2/imgproc.hpp>

#include "VideoExample.h"

struct bounding_box
{
    int x;
    int y;
    int w;
    int h;
};

///
/// \brief DrawFilledRect
///
void DrawFilledRect(cv::Mat& frame, const cv::Rect& rect, cv::Scalar cl, int alpha)
{
	if (alpha)
	{
		const int alpha_1 = 255 - alpha;
		const int nchans = frame.channels();
		int color[3] = { cv::saturate_cast<int>(cl[0]), cv::saturate_cast<int>(cl[1]), cv::saturate_cast<int>(cl[2]) };
		for (int y = rect.y; y < rect.y + rect.height; ++y)
		{
			uchar* ptr = frame.ptr(y) + nchans * rect.x;
			for (int x = rect.x; x < rect.x + rect.width; ++x)
			{
				for (int i = 0; i < nchans; ++i)
				{
					ptr[i] = cv::saturate_cast<uchar>((alpha_1 * ptr[i] + alpha * color[i]) / 255);
				}
				ptr += nchans;
			}
		}
	}
	else
	{
		cv::rectangle(frame, rect, cl, cv::FILLED);
	}
}

///
/// \brief The MotionDetectorExample class
///
class MotionDetectorExample : public VideoExample
{
public:
    MotionDetectorExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser),
          m_minObjWidth(10)
    {
    }

protected:
    ///
    /// \brief InitDetector
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame)
    {
        m_minObjWidth = frame.cols / 50;

        config_t config;
#if 1
        config.emplace("history", std::to_string(cvRound(10 * m_minStaticTime * m_fps)));
        config.emplace("varThreshold", "16");
        config.emplace("detectShadows", "1");
        config.emplace("useRotatedRect", "0");
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Motion_MOG2, config, frame));
#else
        config.emplace("minPixelStability", "15");
        config.emplace("maxPixelStability", "900");
        config.emplace("useHistory", "1");
        config.emplace("isParallel", "1");
        config.emplace("useRotatedRect", "0");
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Motion_CNT, config, frame));
#endif

        if (m_detector.get())
        {
            m_detector->SetMinObjectSize(cv::Size(m_minObjWidth, m_minObjWidth));
            return true;
        }
        return false;
    }
    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
    {
        TrackerSettings settings;
		settings.SetDistance(tracking::DistCenters);
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterCenter;
        settings.m_lostTrackType = tracking::TrackKCF;       // Use visual objects tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.4f;                             // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.5f;                  // Accel noise magnitude for Kalman filter
        settings.m_distThres = 0.95f;                    // Distance threshold between region and object on two frames
        settings.m_minAreaRadius = frame.rows / 20.f;

        settings.m_useAbandonedDetection = false;
        if (settings.m_useAbandonedDetection)
        {
            settings.m_minStaticTime = m_minStaticTime;
            settings.m_maxStaticTime = 60;
            settings.m_maximumAllowedSkippedFrames = cvRound(settings.m_minStaticTime * m_fps); // Maximum allowed skipped frames
            settings.m_maxTraceLength = 2 * settings.m_maximumAllowedSkippedFrames;        // Maximum trace length
        }
        else
        {
            settings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
            settings.m_maxTraceLength = cvRound(4 * m_fps);              // Maximum trace length
        }

        m_tracker = std::make_unique<CTracker>(settings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
		auto tracks = m_tracker->GetTracks();

        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : tracks)
        {
            if (track.m_isStatic)
            {
                DrawTrack(frame, 1, track, true);
            }
            else
            {
                if (track.IsRobust(cvRound(m_fps / 4),          // Minimal trajectory size
                                    0.7f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                    cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                        )
                {
                    DrawTrack(frame, 1, track, true);
                }
            }
        }

        m_detector->CalcMotionMap(frame);
    }

    ///
    /// \brief ZoomInOnROI
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
    cv::Mat ZoomInOnROI(cv::Mat frame, int framesCounter, int currTime)
    {
        if (framesCounter + currTime == 0){
            return frame;
        }
        else
        {
            return frame;
        }
    }
private:
    int m_minObjWidth = 8;
    int m_minStaticTime = 5;
};

// ----------------------------------------------------------------------

///
/// \brief The FaceDetectorExample class
///
class FaceDetectorExample : public VideoExample
{
public:
    FaceDetectorExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitDetector
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame)
    {
#ifdef _WIN32
        std::string pathToModel = "../../data/";
#else
        std::string pathToModel = "../data/";
#endif

        config_t config;
        config.emplace("cascadeFileName", pathToModel + "haarcascade_frontalface_alt2.xml");
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Face_HAAR, config, frame));
        if (m_detector.get())
        {
            m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));
            return true;
        }
        return false;
    }
    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
    {
        TrackerSettings settings;
		settings.SetDistance(tracking::DistJaccard);
        settings.m_kalmanType = tracking::KalmanUnscented;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackCSRT;      // Use visual objects tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                                // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                     // Accel noise magnitude for Kalman filter
        settings.m_distThres = 0.8f;                         // Distance threshold between region and object on two frames
        settings.m_minAreaRadius = frame.rows / 20.f;
        settings.m_maximumAllowedSkippedFrames = cvRound(m_fps / 2);   // Maximum allowed skipped frames
        settings.m_maxTraceLength = cvRound(5 * m_fps);            // Maximum trace length

        m_tracker = std::make_unique<CTracker>(settings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
		auto tracks = m_tracker->GetTracks();

        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : tracks)
        {
            if (track.IsRobust(8,                           // Minimal trajectory size
                                0.4f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, track);
            }
        }

        m_detector->CalcMotionMap(frame);
    }

    ///
    /// \brief ZoomInOnROI
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
    cv::Mat ZoomInOnROI(cv::Mat frame, int framesCounter, int currTime)
    {
        if (framesCounter + currTime == 0){
            return frame;
        }
        else
        {
            return frame;
        }
    }
};

// ----------------------------------------------------------------------

///
/// \brief The PedestrianDetectorExample class
///
class PedestrianDetectorExample : public VideoExample
{
public:
    PedestrianDetectorExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitDetector
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame)
    {
        tracking::Detectors detectorType = tracking::Detectors::Pedestrian_C4; // tracking::Detectors::Pedestrian_HOG;

#ifdef _WIN32
        std::string pathToModel = "../../data/";
#else
        std::string pathToModel = "../data/";
#endif

        config_t config;
        config.emplace("detectorType", (detectorType == tracking::Pedestrian_HOG) ? "HOG" : "C4");
        config.emplace("cascadeFileName1", pathToModel + "combined.txt.model");
        config.emplace("cascadeFileName2", pathToModel + "combined.txt.model_");
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(detectorType, config, frame));
        if (m_detector.get())
        {
            m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));
            return true;
        }
        return false;
    }
    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
    {
        TrackerSettings settings;
		settings.SetDistance(tracking::DistRects);
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackCSRT;   // Use visual objects tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                             // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                  // Accel noise magnitude for Kalman filter
        settings.m_distThres = 0.8f;                      // Distance threshold between region and object on two frames
        settings.m_minAreaRadius = frame.rows / 20.f;
        settings.m_maximumAllowedSkippedFrames = cvRound(m_fps);   // Maximum allowed skipped frames
        settings.m_maxTraceLength = cvRound(5 * m_fps);   // Maximum trace length

        m_tracker = std::make_unique<CTracker>(settings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
		auto tracks = m_tracker->GetTracks();

        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : tracks)
        {
			if (track.IsRobust(cvRound(m_fps / 2),          // Minimal trajectory size
                                0.4f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, track);
            }
        }

        m_detector->CalcMotionMap(frame);
    }

    ///
    /// \brief ZoomInOnROI
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
    cv::Mat ZoomInOnROI(cv::Mat frame, int framesCounter, int currTime)
    {
        if (framesCounter + currTime == 0){
            return frame;
        }
        else
        {
            return frame;
        }
    }
};

// ----------------------------------------------------------------------

///
/// \brief The SSDMobileNetExample class
///
class SSDMobileNetExample : public VideoExample
{
public:
    SSDMobileNetExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitDetector(
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame)
    {
#ifdef _WIN32
        std::string pathToModel = "../../data/";
#else
        std::string pathToModel = "../data/";
#endif
        config_t config;
        config.emplace("modelConfiguration", pathToModel + "MobileNetSSD_deploy.prototxt");
        config.emplace("modelBinary", pathToModel + "MobileNetSSD_deploy.caffemodel");
        config.emplace("confidenceThreshold", "0.5");
        config.emplace("maxCropRatio", "3.0");
        config.emplace("dnnTarget", "DNN_TARGET_CPU");
        config.emplace("dnnBackend", "DNN_BACKEND_INFERENCE_ENGINE");

        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::SSD_MobileNet, config, frame));
        if (m_detector.get())
        {
            m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));
            return true;
        }
        return false;
    }
    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
    {
        TrackerSettings settings;
		settings.SetDistance(tracking::DistRects);
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackCSRT;      // Use visual objects tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                                // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                     // Accel noise magnitude for Kalman filter
        settings.m_distThres = 0.8f;                         // Distance threshold between region and object on two frames
        settings.m_minAreaRadius = frame.rows / 20.f;
        settings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
        settings.m_maxTraceLength = cvRound(5 * m_fps);      // Maximum trace length

        m_tracker = std::make_unique<CTracker>(settings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
		auto tracks = m_tracker->GetTracks();

        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : tracks)
        {
            if (track.IsRobust(5,                           // Minimal trajectory size
                                0.2f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, track);

                std::string label = track.m_type + ": " + std::to_string(track.m_confidence);
                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                cv::Rect brect = track.m_rrect.boundingRect();
				DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(200, 200, 200), 150);
                cv::putText(frame, label, brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }

        m_detector->CalcMotionMap(frame);
    }

    ///
    /// \brief ZoomInOnROI
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
    cv::Mat ZoomInOnROI(cv::Mat frame, int framesCounter, int currTime)
    {
        if (framesCounter + currTime == 0){
            return frame;
        }
        else
        {
            return frame;
        }
    }
};

// ----------------------------------------------------------------------

///
/// \brief The YoloExample class
///
class YoloExample : public VideoExample
{
public:
    YoloExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitDetector(
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame)
    {
        config_t config;
        const int yoloTest = 1;

#ifdef _WIN32
        std::string pathToModel = "../../data/";
#else
        std::string pathToModel = "../data/";
#endif

        switch (yoloTest)
        {
        case 0:
            config.emplace("modelConfiguration", pathToModel + "tiny-yolo.cfg");
            config.emplace("modelBinary", pathToModel + "tiny-yolo.weights");
            break;

        case 1:
            config.emplace("modelConfiguration", pathToModel + "yolov3-tiny.cfg");
            config.emplace("modelBinary", pathToModel + "yolov3-tiny.weights");
            config.emplace("classNames", pathToModel + "coco.names");
            break;
        }

        config.emplace("confidenceThreshold", "0.1");
        config.emplace("maxCropRatio", "2.0");
        config.emplace("dnnTarget", "DNN_TARGET_CPU");
        config.emplace("dnnBackend", "DNN_BACKEND_OPENCV");

        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Yolo_OCV, config, frame));
        if (m_detector.get())
        {
            m_detector->SetMinObjectSize(cv::Size(frame.cols / 40, frame.rows / 40));
            return true;
        }
        return false;
    }
    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
    {
        TrackerSettings settings;
		settings.SetDistance(tracking::DistRects);
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackCSRT;      // Use visual objects tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                                // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.2f;                     // Accel noise magnitude for Kalman filter
        settings.m_distThres = 0.8f;                         // Distance threshold between region and object on two frames
        settings.m_minAreaRadius = frame.rows / 20.f;
        settings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
        settings.m_maxTraceLength = cvRound(5 * m_fps);      // Maximum trace length

        m_tracker = std::make_unique<CTracker>(settings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
		auto tracks = m_tracker->GetTracks();

        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : tracks)
        {
            if (track.IsRobust(1,                           // Minimal trajectory size
                                0.1f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, track);

                std::string label = track.m_type + ": " + std::to_string(track.m_confidence);
                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                cv::Rect brect = track.m_rrect.boundingRect();
				DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(200, 200, 200), 150);
                cv::putText(frame, label, brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }

        m_detector->CalcMotionMap(frame);
    }

    ///
    /// \brief ZoomInOnROI
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
    cv::Mat ZoomInOnROI(cv::Mat frame, int framesCounter, int currTime)
    {
        if (framesCounter + currTime == 0){
            return frame;
        }
        else
        {
            return frame;
        }
    }
};

#ifdef BUILD_YOLO_LIB
// ----------------------------------------------------------------------

///
/// \brief The YoloDarknetExample class
///
class YoloDarknetExample : public VideoExample
{
public:
	YoloDarknetExample(const cv::CommandLineParser& parser)
		:
		VideoExample(parser)
	{
	}

protected:
    ///
    /// \brief InitDetector
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame)
    {
        config_t config;

#ifdef _WIN32
        std::string pathToModel = "../../data/";
#else
        std::string pathToModel = "../data/";
#endif
#if 0
        config.emplace("modelConfiguration", pathToModel + "yolov3-tiny.cfg");
        config.emplace("modelBinary", pathToModel + "yolov3-tiny.weights");
		config.emplace("confidenceThreshold", "0.5");
#else
        config.emplace("modelConfiguration", pathToModel + "yolov3.cfg");
        config.emplace("modelBinary", pathToModel + "yolov3.weights");
		config.emplace("confidenceThreshold", "0.7");
#endif
        config.emplace("classNames", pathToModel + "coco.names");
        config.emplace("maxCropRatio", "-1");

        config.emplace("white_list", "person");
        config.emplace("white_list", "car");
        config.emplace("white_list", "bicycle");
        config.emplace("white_list", "motorbike");
        config.emplace("white_list", "bus");
        config.emplace("white_list", "truck");
        //config.emplace("white_list", "traffic light");
        //config.emplace("white_list", "stop sign");

        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Yolo_Darknet, config, frame));
        if (m_detector.get())
        {
            m_detector->SetMinObjectSize(cv::Size(frame.cols / 40, frame.rows / 40));
            return true;
        }
        return false;
    }

    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
	{
		TrackerSettings settings;
        settings.SetDistance(tracking::DistCenters);
		settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterCenter;
        settings.m_lostTrackType = tracking::TrackKCF;      // Use visual objects tracker for collisions resolving
		settings.m_matchType = tracking::MatchHungrian;
		settings.m_dt = 0.3f;                                // Delta time for Kalman filter
		settings.m_accelNoiseMag = 0.2f;                     // Accel noise magnitude for Kalman filter
        settings.m_distThres = 0.8f;                         // Distance threshold between region and object on two frames
        settings.m_minAreaRadius = frame.rows / 20.f;
		settings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
		settings.m_maxTraceLength = cvRound(5 * m_fps);      // Maximum trace length

		settings.AddNearTypes("car", "bus", false);
		settings.AddNearTypes("car", "truck", false);
		settings.AddNearTypes("person", "bicycle", true);
		settings.AddNearTypes("person", "motorbike", true);

		m_tracker = std::make_unique<CTracker>(settings);

		return true;
	}

    ///
    /// \brief DrawData
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
	void DrawData(cv::Mat frame, int framesCounter, int currTime)
	{
		auto tracks = m_tracker->GetTracks();

		if (m_showLogs)
		{
			std::cout << "Frame " << framesCounter << ": tracks = " << tracks.size() << ", time = " << currTime << std::endl;
		}

		for (const auto& track : tracks)
		{
            if (track.IsRobust(2,                           // Minimal trajectory size
                0.5f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
				cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
				)
			{
				DrawTrack(frame, 1, track);


				std::stringstream label;
				label << track.m_type << " " << std::setprecision(2) << track.m_velocity << ": " << track.m_confidence;
				int baseLine = 0;
				cv::Size labelSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                cv::Rect brect = track.m_rrect.boundingRect();
				if (brect.x < 0)
				{
					brect.width = std::min(brect.width, frame.cols - 1);
					brect.x = 0;
				}
				else if (brect.x + brect.width >= frame.cols)
				{
					brect.x = std::max(0, frame.cols - brect.width - 1);
					brect.width = std::min(brect.width, frame.cols - 1);
				}
				if (brect.y - labelSize.height < 0)
				{
					brect.height = std::min(brect.height, frame.rows - 1);
					brect.y = labelSize.height;
				}
				else if (brect.y + brect.height >= frame.rows)
				{
					brect.y = std::max(0, frame.rows - brect.height - 1);
					brect.height = std::min(brect.height, frame.rows - 1);
				}
				DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(200, 200, 200), 150);
                cv::putText(frame, label.str(), brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
			}
		}

        //m_detector->CalcMotionMap(frame);
	}

    ///
    /// \brief ZoomInOnROI
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
	cv::Mat ZoomInOnROI(cv::Mat frame, int framesCounter, int currTime)
    {
        if (framesCounter + currTime == 0){
            return frame;
        }
        else
        {
            return frame;
        }
    }
};

#endif


// ---------------------------------------------------------------------- //
// ---------------------------------------------------------------------- //
// -------------- Smart Camera Operator Example Settings ---------------- //
// ---------------------------------------------------------------------- //
// ---------------------------------------------------------------------- //

#define bb_history 50

#ifdef BUILD_YOLO_LIB
// ---------------------------------------------------------------------- //

///
/// \brief The SmartCamOperator_w_YoloDarknet class
///
class SmartCamOperator_w_YoloDarknet : public VideoExample
{
public:
	SmartCamOperator_w_YoloDarknet(const cv::CommandLineParser& parser)
		:
		VideoExample(parser)
	{

	}

protected:

    ///
    /// \brief Ring Memory
    ///
    struct bounding_box bb_ring_memory[bb_history];

    int frames_since_init = 0;
    int frames_without_detected_objects = 0;

    ///
    /// \brief InitDetector
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame)
    {
        config_t config;

#ifdef _WIN32
        std::string pathToModel = "../../data/";
#else
        std::string pathToModel = "../data/";
#endif
#if 0
        config.emplace("modelConfiguration", pathToModel + "yolov3-tiny.cfg");
        config.emplace("modelBinary", pathToModel + "yolov3-tiny.weights");
		config.emplace("confidenceThreshold", "0.5");
#else
        config.emplace("modelConfiguration", pathToModel + "yolov3.cfg");
        config.emplace("modelBinary", pathToModel + "yolov3.weights");
		config.emplace("confidenceThreshold", "0.6");
#endif
        config.emplace("classNames", pathToModel + "horse_and_rider.names");
        config.emplace("maxCropRatio", "-1");

        config.emplace("white_list", "Horse&Rider");

        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Yolo_Darknet, config, frame));
        if (m_detector.get())
        {
            m_detector->SetMinObjectSize(cv::Size(frame.cols / 40, frame.rows / 40));
            return true;
        }
        return false;
    }

    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
	{
		TrackerSettings settings;
        settings.SetDistance(tracking::DistCenters);
		settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterCenter;
        settings.m_lostTrackType = tracking::TrackKCF;      // Use visual objects tracker for collisions resolving
		settings.m_matchType = tracking::MatchHungrian;
		settings.m_dt = 0.3f;                                // Delta time for Kalman filter
		settings.m_accelNoiseMag = 0.2f;                     // Accel noise magnitude for Kalman filter
        settings.m_distThres = 0.8f;                         // Distance threshold between region and object on two frames
        settings.m_minAreaRadius = frame.rows / 20.f;
		settings.m_maximumAllowedSkippedFrames = cvRound(.8 * m_fps); // Maximum allowed skipped frames
		settings.m_maxTraceLength = cvRound(2 * m_fps);      // Maximum trace length

		/*
		settings.AddNearTypes("car", "bus", false);
		settings.AddNearTypes("car", "truck", false);
		settings.AddNearTypes("person", "bicycle", true);
		settings.AddNearTypes("person", "motorbike", true);
		*/
		
		m_tracker = std::make_unique<CTracker>(settings);

		// Init RingMemory to show
        for (int i = 0; i < bb_history; i++)
        {
            bb_ring_memory[i] = {0, 0, frame.cols-1, frame.rows-1};
        }

		return true;
	}

    ///
    /// \brief DrawData
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
	void DrawData(cv::Mat frame, int framesCounter, int currTime)
	{
		auto tracks = m_tracker->GetTracks();

		if (m_showLogs)
		{
			std::cout << "Frame " << framesCounter << ": tracks = " << tracks.size() << ", time = " << currTime << std::endl;
		}

		for (const auto& track : tracks)
		{
            if (track.IsRobust(2,                           // Minimal trajectory size
                0.5f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
				cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
				)
			{
				DrawTrack(frame, 1, track);

				std::stringstream label;
				label << track.m_type << " " << std::setprecision(2) << track.m_velocity << ": " << track.m_confidence;
				int baseLine = 0;
				cv::Size labelSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                cv::Rect brect = track.m_rrect.boundingRect();
				if (brect.x < 0)
				{
					brect.width = std::min(brect.width, frame.cols - 1);
					brect.x = 0;
				}
				else if (brect.x + brect.width >= frame.cols)
				{
					brect.x = std::max(0, frame.cols - brect.width - 1);
					brect.width = std::min(brect.width, frame.cols - 1);
				}
				if (brect.y - labelSize.height < 0)
				{
					brect.height = std::min(brect.height, frame.rows - 1);
					brect.y = labelSize.height;
				}
				else if (brect.y + brect.height >= frame.rows)
				{
					brect.y = std::max(0, frame.rows - brect.height - 1);
					brect.height = std::min(brect.height, frame.rows - 1);
				}
				DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(200, 200, 200), 150);
                cv::putText(frame, label.str(), brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
			}
		}

        //m_detector->CalcMotionMap(frame);
	}


	///
    /// \brief ZoomInOnROI
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
	cv::Mat ZoomInOnROI(cv::Mat frame, int framesCounter, int currTime)
	{
		auto tracks = m_tracker->GetTracks();

		if (m_showLogs)
		{
			std::cout << "Frame " << framesCounter << ": tracks = " << tracks.size() << ", time = " << currTime << std::endl;
		}

		if (tracks.size() == 0 && frames_without_detected_objects <= 100)
        {
		    bb_ring_memory[framesCounter % bb_history] = bb_ring_memory[(framesCounter - 1) % bb_history];
            ++frames_without_detected_objects;
        }
		else if (tracks.size() == 0 && frames_without_detected_objects > 100)
        {
		    struct bounding_box entire_frame {0, 0, frame.cols, frame.rows};
		    bb_ring_memory[framesCounter % bb_history] = entire_frame;
        }
        else
        {
            frames_without_detected_objects = 0;
            struct bounding_box joint_ROI {frame.cols, frame.rows, 1, 1};
            for (const auto& track : tracks)
            {
                cv::Rect brect = track.m_rrect.boundingRect();

                if (brect.x < 0)
                {
                    brect.width = std::min(brect.width, frame.cols - 1);
                    brect.x = 0;
                }
                else if ((brect.x + brect.width) >= frame.cols)
                {
                    brect.x = std::max(0, frame.cols - brect.width - 1);
                    brect.width = std::min(brect.width, frame.cols - 1);
                }
                if (brect.y < 0)
                {
                    brect.height = std::min(brect.height, frame.rows - 1);
                    brect.y = 0;
                }
                else if ((brect.y + brect.height) >= frame.rows)
                {
                    brect.y = std::max(0, frame.rows - brect.height - 1);
                    brect.height = std::min(brect.height, frame.rows - 1);
                }
                joint_ROI.x = (brect.x < joint_ROI.x) ?  brect.x : joint_ROI.x;
                joint_ROI.y = (brect.y < joint_ROI.y) ?  brect.y : joint_ROI.y;
                joint_ROI.w = ((brect.x + brect.width) > joint_ROI.w) ? (brect.x + brect.width) : joint_ROI.w;
                joint_ROI.h = ((brect.y + brect.height) > joint_ROI.h) ? (brect.y + brect.height) : joint_ROI.h;

            }
            joint_ROI.w = joint_ROI.w - joint_ROI.x;
            joint_ROI.h = joint_ROI.h - joint_ROI.y;

            struct bounding_box ROI_raw = joint_ROI;


            if (ROI_raw.x - 50 >= 0) ROI_raw.x -= 50;
            if (ROI_raw.y - 50 >= 0) ROI_raw.y -= 50;
            if (ROI_raw.x + ROI_raw.w + 100 <= frame.cols) ROI_raw.w += 100;
            if (ROI_raw.y + ROI_raw.h + 100 <= frame.rows) ROI_raw.h += 100;


            struct bounding_box ROI;

            // check for and extablish 16:9 aspect ratio
            // TODO: This probably needs testing and more case coverage!
            if ((ROI_raw.w / ROI_raw.h) < 1.7777777)
            {
                ROI.y = ROI_raw.y;
                ROI.w = (int) (ROI_raw.h * 1.7777777 + .5);
                ROI.h = ROI_raw.h;

                if ((ROI_raw.x - .5 * (ROI.w - ROI_raw.w)) < 0) {
                    ROI.x = 0;
                    ROI.w -= (int) (ROI_raw.x - .5 * (ROI.w - ROI_raw.w) + .5);
                }
                else if ((ROI_raw.x + ROI_raw.w + .5 * (ROI.w - ROI_raw.w)) >= frame.cols)
                {
                    ROI.x = (int) (ROI_raw.x - (frame.cols - 1 - (ROI_raw.x + .5 * (ROI.w - ROI_raw.w))) - .5);
                    ROI.w = frame.cols - 1;
                }
                else
                {
                    ROI.x = (int) (ROI_raw.x - .5 * (ROI.w - ROI_raw.w) + .5);
                }
            }
            else
            if ((ROI_raw.w / ROI_raw.h) > 1.7777777)
            {
                ROI.x = ROI_raw.x;
                ROI.w = ROI_raw.w;
                ROI.h = (int) (ROI_raw.w * 0.5625 + .5);

                if ((ROI_raw.y - .5 * (ROI.h - ROI_raw.h)) < 0)
                {
                    ROI.y = 0;
                    ROI.h -= (int) ((ROI_raw.y - .5 * (ROI.h - ROI_raw.h)) + .5);
                }
                else if ((ROI_raw.y + ROI_raw.h + .5 * (ROI.h - ROI_raw.h)) >= frame.rows)
                {
                    ROI.y = (int) (ROI_raw.y - (frame.rows - 1 - (ROI_raw.y + .5 * (ROI.h - ROI_raw.h))) - .5);
                    ROI.h = frame.rows - 1;
                }
                else
                {
                    ROI.y = (int) (ROI_raw.y - .5 * (ROI.h - ROI_raw.h) + .5);
                }
            }

            bb_ring_memory[framesCounter % bb_history] = ROI;
        }

        struct bounding_box smoothedROI;
        smoothedROI = expZoomSmoothing(bb_ring_memory, framesCounter, 0.2);

        if (smoothedROI.h == 0 || smoothedROI.w == 0)
        {
            smoothedROI = bb_ring_memory[framesCounter % bb_history];
        }

        cv::Rect crop_region;
        crop_region.x = smoothedROI.x;
        crop_region.y = smoothedROI.y;
        crop_region.width = smoothedROI.w;
        crop_region.height = smoothedROI.h;

        
        if(crop_region.x < 0)
        {
            crop_region.width -= crop_region.x;
            crop_region.x = 0;
        }
        if(crop_region.y < 0)
        {
            crop_region.height -= crop_region.y;
            crop_region.y = 0;
        }
        if(crop_region.x + crop_region.width > frame.cols - 1)
        {
            crop_region.x += frame.cols - 1 - (crop_region.x + crop_region.width);
        }
        if(crop_region.y + crop_region.height > frame.rows - 1)
        {
            crop_region.y += frame.rows - 1 - (crop_region.y + crop_region.height);
        }


        cv::Mat croppedImage = frame(crop_region);
        // Scale ROI to frame size
        cv::resize(croppedImage, croppedImage, cv::Size(frame.cols, frame.rows), 0, 0, cv::INTER_LINEAR);

        return croppedImage;

	}

	//--------------------------------------------------------//
	//------------------- Smoothing --------------------------//
	//--------------------------------------------------------//

	// Source:  https://de.wikibooks.org/wiki/Statistik:_Gl%C3%A4ttungsverfahren

	/**
	 * Exponential Smoothing of Time Series Data, in this case an n-slot ring memory.
	 * Return value should be kept and stored for the next call,
	 * where it should be used as the 'lastSmoothedValue' Parameter.
	 *
	 * @param   alpha               float   Smoothing-Faktor (0,..,1)
	 * @param   lastSmoothedValue   float   Smoothing result for previous entry of time series
	 * @param   value               float   Current value in time series to be smoothed
	 */
	float exp_smoothing (float alpha, float lastSmoothedValue, float value)
	{
	    return alpha * value + (1-alpha) * lastSmoothedValue;
	}

	/**
	 * Double the Exponential Smoothing
	 *
	 * @param   alpha                       float   Smoothing-Faktor (0,..,1)
	 * @param   lastDoublySmoothedValue     float   Doubly smoothed result for previous entry of time series
     * @param   smoothedValue               float   Smoothed value of current entry in time series to be doubly smoothed
	 */
	float doubled_exp_smoothing (float alpha, float lastDoublySmoothedValue, float smoothedValue)
	{
	    return alpha * smoothedValue + (1-alpha) * lastDoublySmoothedValue;
	}

    /**
     * Finish up double smoothing to find a useful prognosis value.
     *
     * @param   smoothedValue       float   Smoothing result for current value in TS
     * @param   doublySmoothedValue float   DoubleSmoothing result for curr value in TS
     */
    float finish_double_smoothing (float smoothedValue, float doublySmoothedValue)
    {
        return std::max(0.0f, 2 * smoothedValue - doublySmoothedValue);
    }

    /**
     * Zoom Smoothing using Exponential Smoothing (double smoothing is used).
     * The purpose of this method is to find reasonable window to crop out of a frame around a ROI;
     * the crop region should be chosen to simulate a camera zoom, while avoiding to be shaky
     * due to inaccurate or moving ROI measurements.
     *
     * @param   array_of_BBs    struct[]     Ring Memory containing a struct (four-touple) holding BB coordinates (ROIs)
     * @param   frame_count     uint        Index of current frame's ROI in ring memory
     * @param   smoothing_fac   float       Smoothing factor alpha, value to be chosen from interval (0,..,1)
     */
    struct bounding_box expZoomSmoothing (struct bounding_box array_of_BBs[], int frame_count, float smoothing_fav)
    {
        // x and y coordinates of a bounding boxes center
        // w and h are width and height of the BB
        float x = 0.0, y = 0.0, w = 0.0, h = 0.0;

        // initialize some values (timeseries needs to start with values furthest backwards)
        struct bounding_box last_smoothed = array_of_BBs[(frame_count - (bb_history - 1)) % bb_history];
        struct bounding_box last_double_smoothed = array_of_BBs[(frame_count - (bb_history - 1)) % bb_history];

        for (int i = bb_history-2; i > 0; --i)
        {
            struct bounding_box curr = array_of_BBs[(frame_count - i) % bb_history];

            last_smoothed.x = exp_smoothing(smoothing_fav, last_smoothed.x, curr.x);
            last_double_smoothed.x = doubled_exp_smoothing(smoothing_fav, last_double_smoothed.x, last_smoothed.x);

            last_smoothed.y = exp_smoothing(smoothing_fav, last_smoothed.y, curr.y);
            last_double_smoothed.y = doubled_exp_smoothing(smoothing_fav, last_double_smoothed.y, last_smoothed.y);

            last_smoothed.w = exp_smoothing(smoothing_fav, last_smoothed.w, curr.w);
            last_double_smoothed.w = doubled_exp_smoothing(smoothing_fav, last_double_smoothed.w, last_smoothed.w);

            last_smoothed.h = exp_smoothing(smoothing_fav, last_smoothed.h, curr.h);
            last_double_smoothed.h = doubled_exp_smoothing(smoothing_fav, last_double_smoothed.h, last_smoothed.h);
        }

        x = finish_double_smoothing(last_smoothed.x, last_double_smoothed.x);
        y = finish_double_smoothing(last_smoothed.y, last_double_smoothed.y);
        w = finish_double_smoothing(last_smoothed.w, last_double_smoothed.w);
        h = finish_double_smoothing(last_smoothed.h, last_double_smoothed.h);

        // eventuell kein finish und stattdessen lastSmoothedValues verwenden (kein doublen)
        // je nach performance kann ein Teil zuvor als Mean berechnet werden und als "kopfelement"
        // in der exponentiellen Glättung verwendet werden. Aber das müssen wir testen.

        return bounding_box {int(x), int(y), int(w), int(h)};
    }
};

#endif
