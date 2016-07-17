#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "MeanShiftTracker.h"

static cv::Mat img;
static cv::Rect window;
static bool selectObject = false;
static int trackObject = 0;

static void onMouse(int event, int x, int y, int, void*)
{
	static cv::Point origin;

	if (selectObject)
	{
		window.x = std::min(x, origin.x);
		window.y = std::min(y, origin.y);
		window.width = std::abs(x - origin.x);
		window.height = std::abs(y - origin.y);

		window &= cv::Rect(0, 0, img.cols, img.rows);
	}

	switch (event)
	{
	case cv::EVENT_LBUTTONDOWN:
		origin = cv::Point(x, y);
		window = cv::Rect(x, y, 0, 0);
		selectObject = true;
		break;
	case cv::EVENT_LBUTTONUP:
		selectObject = false;
		if (window.width > 0 && window.height > 0)
			trackObject = -1;
		break;
	default:
		break;
	}
};

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " input_video" << std::endl;
		return -1;
	}

	cv::VideoCapture cap(argv[1]);
	if (!cap.isOpened())
	{
		std::cerr << argv[1] << "could not be opened." << std::endl;
		return -1;
	}

	MeanShiftTracker tracker;
	cv::Mat frame;
	bool backprojMode = false;
	bool paused = false;

	cv::namedWindow("MeanShiftTracker Demo");
	cv::setMouseCallback("MeanShiftTracker Demo", onMouse);

	while (true)
	{
		if (!paused) {
			cap >> frame;
			if (frame.empty())
				break;
		}

		frame.copyTo(img);

		if (!paused) {

			if (trackObject < 0)
			{
				tracker.start(img, window);
				trackObject = 1;
			}

			if (trackObject > 0)
			{
				tracker.update(img, window);

				if (backprojMode)
				{
					cv::cvtColor(tracker.backProject_, img, cv::COLOR_GRAY2BGR);
				}
				cv::rectangle(img, window.tl(), window.br(), cv::Scalar(0, 0, 255), 2);
			}

		}
		else if (trackObject < 0)
			paused = false;

		if (selectObject && window.width > 0 && window.height > 0)
		{
			cv::Mat roi(img, window);
			cv::bitwise_not(roi, roi);
		}

		cv::imshow("MeanShiftTracker Demo", img);

		char c = cv::waitKey(25);
		if (c == 27)
			break;

		switch (c)
		{
		case 'b':
			backprojMode = !backprojMode;
			break;
		case 't':
			trackObject = 0;
			break;
		case 'p':
			paused = !paused;
			break;
		default:
			break;
		}
	}

	return 0;
}
