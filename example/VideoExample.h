#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <atomic>

#include "BaseDetector.h"
#include "Ctracker.h"

///
/// \brief The VideoExample class
///
class VideoExample
{
public:
    VideoExample(const cv::CommandLineParser& parser);
    VideoExample(const VideoExample&) = delete;
    VideoExample(VideoExample&&) = delete;
    VideoExample& operator=(const VideoExample&) = delete;
    VideoExample& operator=(VideoExample&&) = delete;

    virtual ~VideoExample() = default;

    void AsyncProcess();
    void AsyncProcess_SmartCamOp();
    void SyncProcess();

    cv::UMat getFrame()
    {
        return this->m_frameInfo[0].m_frame.getUMat(cv::ACCESS_READ);
    }

protected:
    std::unique_ptr<BaseDetector> m_detector;
    std::unique_ptr<CTracker> m_tracker;

    bool m_showLogs = true;
    float m_fps = 25;

    int m_captureTimeOut = 60000;
    int m_trackingTimeOut = 60000;

    static void CaptureAndDetect(VideoExample* thisPtr, std::atomic<bool>& stopCapture);
    static void CaptureAndDetect_SmartCamOp(VideoExample* thisPtr, std::atomic<bool>& stopCapture);

    virtual bool InitDetector(cv::UMat frame) = 0;
    virtual bool InitTracker(cv::UMat frame) = 0;

    void Detection(cv::Mat frame, regions_t& regions);
    void Detection_SmartCamOp(cv::Mat frame, regions_t& regions, int frameCounter);
    void Tracking(cv::Mat frame, const regions_t& regions);

    virtual void DrawData(cv::Mat frame, int framesCounter, int currTime) = 0;

    virtual cv::Mat ZoomInOnROI(cv::Mat frame, int framesCounter, int currTime) = 0;

    void DrawTrack(cv::Mat frame, int resizeCoeff, const TrackingObject& track, bool drawTrajectory = true);

private:
    bool m_isTrackerInitialized = false;
    bool m_isDetectorInitialized = false;
    std::string m_inFile;
    std::string m_outFile;
    std::string m_outFileZoomed;
    int m_fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    int m_startFrame = 0;
    int m_endFrame = 0;
    int m_finishDelay = 0;
    std::vector<cv::Scalar> m_colors;

    struct FrameInfo
    {
        cv::Mat m_frame;
        regions_t m_regions;
        int64 m_dt = 0;

        std::condition_variable m_cond;
        std::mutex m_mutex;
        bool m_captured = false;
    };
    FrameInfo m_frameInfo[2];

    bool OpenCapture(cv::VideoCapture& capture);
    bool WriteFrame(cv::VideoWriter& writer, const cv::Mat& frame);
    bool WriteZoomedFrame(cv::VideoWriter& writer, const cv::Mat& frame);
};
