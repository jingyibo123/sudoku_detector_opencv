/*
 * Detector.h
 *
 *  Created on: 19 sept. 2017
 *      Author: JING Yibo
 */

#ifndef SRC_DETECTOR_H_
#define SRC_DETECTOR_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <string>



#define RSC_PATH "../../../Workspace/resource/"
#define SVM_NAME "cppsvm.yml.gz"
#define SVM_PATH "../../../Workspace/resource/cppsvm.yml.gz"
#define DIGIT_MASK_PATH "../../../Workspace/resource/digitmasksRev3.png"


#define SZ 9
#define SZ_IMG 28
#define NB_BIN 16
#define NB_AREA 25

// ----Parameters of finding grid
#define PARAM_GRID_BLUR 3
#define PARAM_GRID_THRES_BLOCKSIZE 11
#define PARAM_GRID_THRES_C 2
// denoise
#define PARAM_GRID_OPEN_KERNEL MORPH_ELLIPSE
#define PARAM_GRID_OPEN_KERNEL_SIZE 2
// fill holes
#define PARAM_GRID_CLOSE_KERNEL MORPH_ELLIPSE
#define PARAM_GRID_CLOSE_KERNEL_SIZE 5
// Threshold of size proportion of grid of the whole image
#define PARAM_GRID_SIZE_THRES 1.0/25

// ----Parameters of locating digit
#define PARAM_CELL_BLUR 7
#define PARAM_CELL_THRES_BLOCKSIZE 15
#define PARAM_CELL_THRES_C 2
// denoise
#define PARAM_CELL_OPEN_KERNEL MORPH_ELLIPSE
#define PARAM_CELL_OPEN_KERNEL_SIZE 2


// ERROR CODES
#define DETECTOR_EMPTY_INPUT_IMG 11
#define DETECTOR_NO_GRID_FOUND 21
#define DETECTOR_NO_DIGIT_IN_CELL 31
#define DETECTOR_FAILED_TO_LOCATE_DIGIT 41


class Detector {
public:
	Detector(cv::Mat img);
	~Detector();
	static bool initiated;
	static void init(const char* svmPath, const char* digitsMaskPath);


	std::vector<cv::Point> pts;
	std::vector<int> extractedDigits;

	int detectPuzzle();

	cv::Mat drawDigits(std::vector<int> resolvedDigits);

	static void deskew(cv::Mat &img, cv::Mat &deskew);
	static void hog(cv::Mat &img, std::vector<float> &hists);

private:
	cv::Mat img;
	bool timerDebug;

    static std::vector<cv::Mat> digitImgs;
	static cv::Ptr<cv::ml::SVM> svm;

	int locatePuzzle(cv::Mat &gray, std::vector<cv::Point> &pts);
	int locateDigit(cv::Mat cell_thresh, cv::Mat &digit);

	static cv::Mat addBorder(cv::Mat &digit, int sampleHeight, int sampleWidth, int sampleDigitHeight);
	static void warpPerspectiveGrid(std::vector<cv::Point> pts, cv::Mat img, cv::Mat &grid);
	static void drawReversePerspectiveGrid(std::vector<cv::Point> pts, cv::Mat &img, cv::Mat grid);

};
#endif /* SRC_DETECTOR_H_ */
