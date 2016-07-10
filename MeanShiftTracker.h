#pragma once

#include <opencv2/opencv.hpp>

class MeanShiftTracker
{
public:
	MeanShiftTracker();
	void start(const cv::Mat& img, const cv::Rect& window);
	int update(const cv::Mat& img, cv::Rect& window);
	int vmin_, vmax_, smin_;
	cv::Rect window_;
	cv::Mat hist_, backProject_;
};
