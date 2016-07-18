#include "MeanShiftTracker.h"

namespace {

	void calcHist(const cv::Mat& img, const cv::Mat& mask, cv::Mat& hist, int histSize, const float *range)
	{
		CV_Assert(img.type() == CV_8U);
		CV_Assert(mask.type() == CV_8U);
		CV_Assert(mask.size() == img.size());

		hist.create(cv::Size(1, histSize), CV_32F);
		hist = cv::Scalar::all(0);

		float minv = range[0];
		float maxv = range[1];

		for (int i = 0; i < img.rows; ++i)
		{
			for (int j = 0; j < img.cols; ++j)
			{
				float v = img.at<uchar>(i, j);
				if (v < minv || v >= maxv)
					continue;

				float nv = (v - minv) / (maxv - minv);
				int bin = static_cast<int>(histSize * nv);
				CV_Assert(bin < histSize);
				if (mask.at<uchar>(i, j))
					hist.at<float>(bin) += 1.0f;
			}
		}
	}

	void calcBackProject(const cv::Mat& img, const cv::Mat& hist, cv::Mat& backProject, const float *range)
	{
		CV_Assert(img.type() == CV_8U);
		CV_Assert(hist.type() == CV_32F);

		backProject.create(img.size(), CV_8U);
		backProject = cv::Scalar::all(0);

		cv::Mat _hist = hist;
		cv::normalize(_hist, _hist, 0, 255, cv::NORM_MINMAX);

		int histSize = hist.rows;
		float minv = range[0];
		float maxv = range[1];

		for (int i = 0; i < img.rows; ++i)
		{
			for (int j = 0; j < img.cols; ++j)
			{
				float v = img.at<uchar>(i, j);
				if (v < minv || v >= maxv)
					continue;

				float nv = (v - minv) / (maxv - minv);
				int bin = static_cast<int>(histSize * nv);
				CV_Assert(bin < histSize);
				backProject.at<uchar>(i, j) = cv::saturate_cast<uchar>(_hist.at<float>(bin));
			}
		}
	}

	int meanShift(const cv::Mat& probImage, cv::Rect& window)
	{
		CV_Assert(probImage.type() == CV_8U);

		int maxiter = 20;
		for (int iter = 0; iter < maxiter; ++iter)
		{
			// 重心の計算
			unsigned int mz = 0, mx = 0, my = 0;
			for (int i = 0; i < window.height; ++i)
			{
				for (int j = 0; j < window.width; ++j)
				{
					int y = i + window.y;
					int x = j + window.x;

					mz += probImage.at<uchar>(y, x);
					mx += x * probImage.at<uchar>(y, x);
					my += y * probImage.at<uchar>(y, x);
				}
			}

			if (mz == 0)
				return 0;

			int cx = mx / mz;
			int cy = my / mz;

			// ウィンドウの位置を更新
			int winx = cx - window.width/2;
			int winy = cy - window.height/2;

			winx = std::min(probImage.cols - 1 - window.width, std::max(0, winx));
			winy = std::min(probImage.rows - 1 - window.height, std::max(0, winy));

			// 移動量が小さい場合は終了
			if (abs(winx - window.x) < 1 && abs(winy - window.y) < 1)
				break;

			window.x = winx;
			window.y = winy;
		}

		return 1;
	}

}

MeanShiftTracker::MeanShiftTracker()
{
	vmin_ = 10;
	vmax_ = 256;
	smin_ = 30;
}

void MeanShiftTracker::start(const cv::Mat& img, const cv::Rect& window)
{
	CV_Assert(img.type() == CV_8UC3);

	// ROIの設定
	cv::Mat roi = img(window);

	// HSVに変換
	cv::Mat hsv;
	cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

	// マスクの作成
	cv::Mat mask;
	cv::inRange(hsv, cv::Scalar(0, smin_, vmin_), cv::Scalar(180, 256, vmax_), mask);

	// Hue成分の抽出
	cv::Mat hue(hsv.size(), hsv.depth());
	int fromTo[2] = { 0, 0 };
	cv::mixChannels({ hsv }, { hue }, fromTo, 1);

	// ヒストグラムの計算
	int histSize = 64;
	float range[] = { 0, 180 };
	calcHist(hue, mask, hist_, histSize, range);
}
	
int MeanShiftTracker::update(const cv::Mat& img, cv::Rect& window)
{
	// HSVに変換
	cv::Mat hsv;
	cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

	// マスクの作成
	cv::Mat mask;
	cv::inRange(hsv, cv::Scalar(0, smin_, vmin_), cv::Scalar(180, 256, vmax_), mask);

	// Hue成分の抽出
	cv::Mat hue(hsv.size(), hsv.depth());
	int fromTo[2] = { 0, 0 };
	cv::mixChannels({ hsv }, { hue }, fromTo, 1);

	// ヒストグラムの逆投影
	float range[] = { 0, 180 };
	calcBackProject(hue, hist_, backProject_, range);

	// 逆投影画像のマスキング
	cv::bitwise_and(backProject_, mask, backProject_);

	// ウィンドウの更新
	return meanShift(backProject_, window);
}