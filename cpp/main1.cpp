
#include <iostream>
#include <fstream>
#include <stdio.h>
#include "Detector.h"
#include "VideoStarter.h"
#include "Timer.cpp"
#include "Eval.cpp"
#include "Puzzle.h"
#include "Train.cpp"



int main(int, char**)
{
	Timer timer("main", true);
	Detector::init(SVM_PATH, DIGIT_MASK_PATH);
	Puzzle::init();
	timer.timeit("init");
//	 Train::trainSvm();

	cv::Mat image = cv::imread("../../../Workspace/resource/images/all/image1050.jpg");
	if (image.empty())                      // Check for invalid input
	{
		std::cout << "Could not open or find the image" << std::endl;
		return 2;
	}
	timer.timeit("read image");
	Detector d = Detector(image);
	timer.timeit("initialize detector");
	std::vector<int> extractedDigits(81);
	d.detectPuzzle();

	timer.timeit("extract puzzle");
	std::vector<int> solved;
	Puzzle puzzle(extractedDigits.data());
	timer.timeit("solve puzzle");
	if(!puzzle.solved){
		std::cout << "puzzle unsolvable";
		// puzzle.disp();
	} else {
		solved = puzzle.getResolvedDigits();
	}

//	int p = 3;


	/* Test Eval  */
//	Eval::evalOne("image153"); // easy
//	Eval::evalOne("image103");// small img, easy
//	Eval::evalOne("image1086");
//	Eval::evalAll();


	/* Test Puzzle class */
//    const char* yibo = "100070030830600000002900608600004907090000050307500004203009100000002043040080009";
//    const char* novig = "003020600900305001001806400008102900700000008006708200002609500800203009005010300";
//    const char* hard = "4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......";
//    const char* bug = "....14....3....2...7..........9...3.6.1.............8.2.....1.4....5.6.....7.8...";
//    const char* bug2 = "002009105080703900000000020050040000700908003000010060040000000008201090903800600";
//	Puzzle puzzle(yibo);
//	if(!puzzle.solve())
//		cout << "solved" << "\n";
//	for(int i : puzzle.getResolvedDigits()){
//		if (i == 0){
//			cout << "not completely solved" << "\n";
//			break;
//		}
//	}



//	ifstream file("../../resource/puzzles.txt");
//	string line;
//	timer.timeit("init");
//	int i = 0;
//	while (getline(file, line)) {
//		if(line.size() < 10)
//			break;
//		Puzzle puzzle(line.data());
//		int re = puzzle.solve();
//		if(re == PUZZLE_SOLVED){
//			i++;
//			// puzzle.disp();
//		} else if (re == PUZZLE_NOT_ENOUGH_DIGITS){
//			cout << "not enough digits" << "\n";
//		} else if (re == PUZZLE_NO_SOLUTION){
//			cout << "puzzle not solveable" << "\n";
//		}
//	}
//	cout << to_string(i) << " puzzles solved \n";


//	VideoStarter::startVideo();

	/* test import tf model

//	Net net = readNetFromTensorflow("../../resource/saved_model1/saved_model.pb");
	Net net = readNetFromTensorflow("../../resource/save1/model2.pb");

//	Ptr<Importer> importer = createTensorflowImporter("../../resource/model1.pb");



	Mat img = imread("../../resource/extractedDigits/8/fromimage164_digit8_pos22.png");
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	gray = gray.reshape(1, 1);
	bitwise_not(gray, gray);
//	Mat grayF(1, 28*28, CV_32FC1);
	gray.convertTo(gray, CV_32F);
	gray = gray / 255.0;

	float* grayFData = (float*) gray.data;

	Mat grayF(1, 28*28, CV_32F);
//	Mat input = blobFromImage(grayF); // no effect still bugging
	Mat input(1, 28*28, CV_32F, (uchar*)grayFData);
	net.setInput(input, "Placeholder_1");

	Mat out = net.forward();
	std::cout << "final----- " << out.dims << std::endl;

	*/

	timer.timeall("main process");
    return 0;
}
