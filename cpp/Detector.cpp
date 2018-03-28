/*
 * Detector.cpp
 *
 *  Created on: 19 sept. 2017
 *      Author: E507067
 */

#include <iostream>
#include <string>
#include "Detector.h"
#include "Timer.cpp"



bool Detector::initiated = false;
std::vector<cv::Mat> Detector::digitImgs;
cv::Ptr<cv::ml::SVM> Detector::svm;


Detector::Detector(cv::Mat img) {
	this->img = img;
	this->timerDebug = true;
}

Detector::~Detector() {

}

void Detector::init(const char* svmPath, const char* digitsMaskPath){
	if(initiated)
		return;

	// load svm machine
	svm = cv::Algorithm::load<cv::ml::SVM>(svmPath);
	// load digit masks
	digitImgs = std::vector<cv::Mat>(SZ);
	cv::Mat digitMasks = cv::imread(digitsMaskPath, CV_LOAD_IMAGE_UNCHANGED);
	for (int i = 0; i < SZ; i++) {
		digitImgs[i] = digitMasks(cv::Rect(0, i * 80, 80, 80));
	}

	initiated = true;
}

int Detector::detectPuzzle() {
	if(img.empty())
		return DETECTOR_EMPTY_INPUT_IMG;

	Timer timer = Timer("detect puzzle", timerDebug);
	timer.timeit();

	cv::Mat gray;
    if (img.channels() == 3) {
    	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
    	cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    }

	timer.timeit("cv::cvtColor to gray");

	if (locatePuzzle(gray, pts)) {
		// no grid found
		return DETECTOR_NO_GRID_FOUND;
	}

	timer.timeit("find grid");

	cv::Mat grayGrid;
	warpPerspectiveGrid(pts, gray, grayGrid);

	timer.timeit("perspective transform");

	double cellh = grayGrid.rows * 1.0 / SZ;
	double cellw = grayGrid.cols * 1.0 / SZ;

	int nb_empty = 0;

	std::vector<int> iCells;
	std::vector<cv::Mat> digits;

	timer.timeit();

	cv::Mat grid_blur;
	cv::GaussianBlur(grayGrid, grid_blur, cv::Size(7, 7), 0);
//	bilateralFilter(grayGrid, grid_blur, 15, 60, 60);   // TOO SLOW!!!  60+ ms per img

	cv::Mat grid_thresh;
	// TODO try ADAPTIVE_THRESH_MEAN_C (little performance gain)
	cv::adaptiveThreshold(grid_blur, grid_thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
			cv::THRESH_BINARY, 15, 2);

	// Use morphology to remove noise
	cv::Mat kernal = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
	morphologyEx(grid_thresh, grid_thresh, cv::MORPH_OPEN, kernal);

	timer.timeit("grid thres");

	for (int n = 0; n < SZ * SZ; n++) {

		cv::Mat cell = grid_thresh(cv::Rect(n % SZ * cellw, n / SZ * cellh, cellw, cellh));

		cv::Mat digit;
		if (locateDigit(cell, digit)) {
			nb_empty++;
			continue;
		}
		iCells.push_back(n);
		digits.push_back(digit);
	}

	timer.timeit("locate digit");

	std::vector<cv::Mat> filledDigits;
	for (unsigned int n = 0; n < iCells.size(); n++) {

		cv::Mat filledDigit = addBorder(digits[n], 28, 28, 28);
		filledDigits.push_back(filledDigit);
	}

	timer.timeit("add border");

	std::vector<int> results;
	for (unsigned int n = 0; n < iCells.size(); n++) {

		std::vector<float> hogData(NB_AREA * NB_BIN);
		hog(filledDigits[n], hogData);

		cv::Mat testData = cv::Mat(1, NB_AREA * NB_BIN, CV_32F, hogData.data());

		float re = svm->predict(testData);
		results.push_back((int) re);
	}
	std::vector<int> extractedDigits(81);
	for (unsigned int n = 0; n < iCells.size(); n++) {
		extractedDigits[iCells[n]] = results[n];
	}
	this->extractedDigits = extractedDigits;

	timer.timeit("hog & predict");

	timer.timeall("all");

	return 0;

}

int Detector::locatePuzzle(cv::Mat &gray, std::vector<cv::Point> &pts) {
	Timer timer = Timer("locate grid", timerDebug);
	// blur to eliminate noise
	cv::Mat blur;
	cv::GaussianBlur(gray, blur, cv::Size(PARAM_GRID_BLUR, PARAM_GRID_BLUR), 0);
//	bilateralFilter(gray, blur, 15, 60, 60);   // TOO SLOW!!!  60+ ms per img

	timer.timeit("blur");

	cv::Mat thresh;
	// TODO use ADAPTIVE_THRESH_MEAN_C to increase speed
	cv::adaptiveThreshold(gray, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
			cv::THRESH_BINARY_INV, PARAM_GRID_THRES_BLOCKSIZE, PARAM_GRID_THRES_C);

	timer.timeit("threshold");

	// Use morphology to remove noise
	cv::Mat kernel = cv::getStructuringElement(cv::PARAM_GRID_OPEN_KERNEL,
			cv::Size(PARAM_GRID_OPEN_KERNEL_SIZE, PARAM_GRID_OPEN_KERNEL_SIZE));
	cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kernel);

	// Fill possible holes in lines
	cv::Mat kernel2 = cv::getStructuringElement(cv::PARAM_GRID_CLOSE_KERNEL,
			cv::Size(PARAM_GRID_CLOSE_KERNEL_SIZE, PARAM_GRID_CLOSE_KERNEL_SIZE));
	cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, kernel2);

	timer.timeit("morphology");


	// TODO use RETR_EXTERNAL to better accuracy and accelerate (5800->5500), but loses one grid (cascade grids?)
	// TODO use hierarchy to filer, accelerate.
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(thresh, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	timer.timeit("find coutours");

	std::map<double, std::vector<cv::Point>> grids;

	for (size_t i = 0; i < contours.size(); i++) {

		double area = contourArea(contours[i]);
		if (area > gray.total() * PARAM_GRID_SIZE_THRES) {
			double peri = arcLength(contours[i], true);
			std::vector<cv::Point> approx;
			approxPolyDP(contours[i], approx, peri * 0.04, true);
			if (approx.size() == 4) {
				grids[area] = approx;
			}
		}
	}

	if (grids.empty())
		return DETECTOR_NO_GRID_FOUND;
	// iterate from the largest grid to the smallest
	for (auto gridIt = grids.rbegin(); gridIt != grids.rend(); ++gridIt) {
		std::vector<cv::Point>grid = gridIt->second;
		// sort four vertexes to order (left-top, right-top, right-bottom, left-bottom)
		sort(grid.begin(), grid.end(), [ ]( const cv::Point& p1, const cv::Point& p2 ) {
			return p1.x + p1.y < p2.x + p2.y;
		});

		// right-bottom should be 3rd instead of 4th
		cv::Point p3 = grid[2];
		grid[2] = grid[3];
		grid[3] = p3;
		if (grid[1].x < grid[3].x) {
			cv::Point p1 = grid[1];
			grid[1] = grid[3];
			grid[3] = p1;
		}

		// see if parallel sides are of similar length
		float thres_diff_length = 0.5;
		float l1 = (grid[1].x - grid[0].x) * (grid[1].x - grid[0].x)
				+ (grid[1].y - grid[0].y) * (grid[1].y - grid[0].y);

		float l2 = (grid[2].x - grid[1].x) * (grid[2].x - grid[1].x)
				+ (grid[2].y - grid[1].y) * (grid[2].y - grid[1].y);

		float l3 = (grid[3].x - grid[2].x) * (grid[3].x - grid[2].x)
				+ (grid[3].y - grid[2].y) * (grid[3].y - grid[2].y);

		float l4 = (grid[0].x - grid[3].x) * (grid[0].x - grid[3].x)
				+ (grid[0].y - grid[3].y) * (grid[0].y - grid[3].y);

		if (l1 / l3 > 1 - thres_diff_length && l1 / l3 < 1 + thres_diff_length
				&& l2 / l4 > 1 - thres_diff_length
				&& l2 / l4 < 1 + thres_diff_length) {
			pts = grid;

			timer.timeit("filter coutours");

			return 0;
		}
	}

	timer.timeit("filter coutours");

	return DETECTOR_NO_GRID_FOUND;

}

/**  returns 0 if digit located, 1 if no digit, 2 if failed to locate **/
int Detector::locateDigit(cv::Mat cell_thresh, cv::Mat &digit) {
	int h = cell_thresh.rows;
	int w = cell_thresh.cols;

	// count the active pixels in the center to see if it contains a digit
	cv::Mat thresh_center = cell_thresh(cv::Rect(0.25 * w, 0.25 * h, 0.5 * w, 0.5 * h));

	int nbActivePixels = countNonZero(thresh_center);

	if (1.0 * nbActivePixels / thresh_center.total() > 0.85) {
		return DETECTOR_NO_DIGIT_IN_CELL;
	}

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(cell_thresh, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	double maxRadius = 0;
	std::vector<cv::Point> digitContour;

	for (std::vector<cv::Point> contour : contours) {

		cv::Point2f center;

		float radius;
		minEnclosingCircle(contour, center, radius);

		if (radius < w * 0.45 && radius > w * 0.20 && radius > maxRadius
				&& (center.y - h / 2) * (center.y - h / 2) < h * h / 16
				&& (center.x - w / 2) * (center.x - w / 2) < w * w / 16) {

			maxRadius = radius;
			digitContour = contour;
		}
	}

	if (digitContour.size() <= 0) {
		return DETECTOR_FAILED_TO_LOCATE_DIGIT;
	}

	cv::Rect rect = cv::boundingRect(digitContour);
	// mask to eliminate noise in digit's rect  overhead: 6.7->7.0
	cv::Mat mask = cv::Mat(cell_thresh.size(), CV_8UC1, cv::Scalar(255));
	std::vector<std::vector<cv::Point>> digitContours;
	digitContours.push_back(digitContour);

	cv::drawContours(mask, digitContours, 0, 0, cv::FILLED);

	bitwise_or(cell_thresh, mask, cell_thresh);

	cell_thresh(rect).copyTo(digit);

	return 0;
}

void Detector::warpPerspectiveGrid(std::vector<cv::Point> pts, cv::Mat img, cv::Mat &grid) {

	int gridRows = (int) (pts[2].y + pts[3].y - pts[0].y - pts[1].y) / 2;
	int gridCols = (int) (pts[1].x + pts[2].x - pts[0].x - pts[3].x) / 2;

	std::vector<cv::Point2f> p = { (cv::Point2f) pts[0], (cv::Point2f) pts[1], (cv::Point2f) pts[2],
			(cv::Point2f) pts[3] };

	std::vector<cv::Point2f> h = { cv::Point2f(0, 0), cv::Point2f(gridCols - 1, 0), cv::Point2f(
			gridCols - 1, gridRows - 1), cv::Point2f(0, gridRows - 1) };

	cv::Mat m = getPerspectiveTransform(p, h);

	cv::warpPerspective(img, grid, m, cv::Size(gridCols, gridRows));

}

/**
Draw the extracted grid back to the image with reverse perspective

@param pts the four corners of the position projected grid on the img
@param img the original image to draw onto
@param grid the grid image to be drawed
 */
void Detector::drawReversePerspectiveGrid(std::vector<cv::Point> pts, cv::Mat &img,
		cv::Mat grid) {
	int gridRows = grid.rows;
	int gridCols = grid.cols;
	std::vector<cv::Point2f> p = { (cv::Point2f) pts[0], (cv::Point2f) pts[1], (cv::Point2f) pts[2],
			(cv::Point2f) pts[3] };

	std::vector<cv::Point2f> h = { cv::Point2f(0, 0), cv::Point2f(gridCols - 1, 0), cv::Point2f(
			gridCols - 1, gridRows - 1), cv::Point2f(0, gridRows - 1) };

	cv::Mat m = getPerspectiveTransform(p, h);

	cv::warpPerspective(grid, img, m, img.size(), cv::WARP_INVERSE_MAP,
			cv::BORDER_TRANSPARENT, cv::Scalar(0.0));
}

cv::Mat Detector::addBorder(cv::Mat &digit, int sampleHeight,
		int sampleWidth, int sampleDigitHeight) {

	if (sampleDigitHeight > sampleHeight) {
		return digit;
	}

	int h = digit.rows;
	int w = digit.cols;

	int sampleDigitWidth = w * sampleDigitHeight / h;
	// detected digit is too wide
	if (sampleDigitWidth > sampleWidth) {
		sampleDigitWidth = sampleWidth;
		sampleDigitHeight = h * sampleDigitWidth / w;
	}

	cv::Mat newDigit;
	cv::resize(digit, newDigit, cv::Size(sampleDigitWidth, sampleDigitHeight), 0, 0,
			cv::INTER_AREA);

	cv::Mat filledDigit(cv::Size(sampleHeight, sampleWidth), CV_8UC1, cv::Scalar(255));

	int x = (sampleWidth - sampleDigitWidth) / 2;
	int y = (sampleHeight - sampleDigitHeight) / 2;

	newDigit.copyTo(filledDigit(cv::Rect(x, y, newDigit.cols, newDigit.rows)));
	return filledDigit;
}

void Detector::hog(cv::Mat &img, std::vector<float> &hists) {
	cv::Mat gx;
	cv::Mat gy;
	Sobel(img, gx, CV_32F, 1, 0);
	Sobel(img, gy, CV_32F, 0, 1);

	cv::Mat mag;
	cv::Mat ang;
	cartToPolar(gx, gy, mag, ang);

	ang = ang * (NB_BIN - 1) / (2 * 3.1415926535897932384626433832795);
	cv::Mat bins;
	ang.convertTo(bins, CV_8U);

	int lines[] = { 0, 5, 11, 16, 22, 28 };
	int nb_col = 5;

	for (int c = 0; c < nb_col; c++) {

		for (int r = 0; r < nb_col; r++) {
			// System.out.println(r + ", " + c);
			cv::Mat subBins = bins(
					cv::Rect(lines[c], lines[r], lines[c + 1] - lines[c],
							lines[r + 1] - lines[r]));
			unsigned char* areaBins = (unsigned char*) subBins.data; // size: subBins.total()

			cv::Mat subMag = mag(
					cv::Rect(lines[c], lines[r], lines[c + 1] - lines[c],
							lines[r + 1] - lines[r]));
			float* areaMag = (float*) subMag.data; // size: subMag.total()

			for (unsigned int i = 0; i < subMag.total(); i++) {

				hists[(c * nb_col + r) * NB_BIN + areaBins[i]] += areaMag[i];
			}
		}
	}
}

void Detector::deskew(cv::Mat &img, cv::Mat &deskew) {
	cv::Moments moms = moments(img);
	 if (moms.mu02 < 1e-2 && moms.mu02 > -1e-2){
		 img.copyTo(deskew);
		 return;
	 }
	 double skew = moms.mu11 / moms.mu02;

	 float mf[2][3] = {{1, skew, (float) (-0.5*SZ_IMG*skew)}, {0, 1, 0}};

	 cv::Mat m = cv::Mat(2, 3, CV_32F, mf);

	 cv::warpAffine(img, deskew, m, cv::Size(SZ_IMG, SZ_IMG), cv::WARP_INVERSE_MAP | cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255));

}

cv::Mat Detector::drawDigits(std::vector<int> resolvedDigits) {
	if (resolvedDigits.size() != SZ * SZ || extractedDigits.size() != SZ * SZ) {
		return img;
	}

	Timer timer = Timer("draw digits", timerDebug);

	int imgChn = img.channels();
	if (imgChn == 4) {
		cv::cvtColor(img, img, cv::COLOR_BGRA2BGR);
	}

	cv::Mat grid;
	warpPerspectiveGrid(pts, img, grid);
	timer.timeit("perspective");

	// draw resolved digits
	for (int i = 0; i < SZ * SZ; i++) {


		int cellh = grid.rows / SZ;
		int cellw = grid.cols / SZ;

		cv::Mat cell = grid(
				cv::Rect((i % SZ) * grid.cols / SZ, (i / SZ) * grid.rows / SZ,
						cellw, cellh));

		// extracted digits
		if (extractedDigits[i] != 0){
			cv::Mat mask;
			cv::resize(digitImgs[extractedDigits[i] - 1], mask, cell.size());

			cv::multiply(mask, cell, mask);

			cv::addWeighted(mask, 0.2, cell, 0.8, 0.0, cell);
			continue;
		}

		// resolved digits
		if (extractedDigits[i] == 0 && resolvedDigits[i] != 0){
			cv::Mat mask;
			cv::resize(digitImgs[resolvedDigits[i] - 1], mask, cell.size());

			cv::multiply(mask, cell, mask);

			cv::addWeighted(mask, 0.6, cell, 0.4, 0.0, cell);
		}

	}

	timer.timeit("drawn cells");
	// do reverse perspective transforcv::Mation and draw the grid with digits onto original image
	drawReversePerspectiveGrid(pts, img, grid);

	timer.timeit("reverse perspective");
	if (imgChn == 4) {
		cv::cvtColor(img, img, cv::COLOR_BGR2BGRA);
	}
//	cv::imwrite("source/drawn.jpg", input);
	return img;
}
