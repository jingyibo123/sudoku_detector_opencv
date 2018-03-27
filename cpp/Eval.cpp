/*
 * Timeit.cpp
 *
 *  Created on: 20 sept. 2017
 *      Author: JING Yibo
 */

#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include "Timer.cpp"
#include "Detector.h"
#include "Puzzle.h"



#define IMG_PATH "../../Workspace/resource/images/"
#define TEST_FILE_LIST_PATH "../../Workspace/resource/testFiles.txt"


class Eval {
public:
	Eval() {
	}

	~Eval() {

	}

	static std::vector<int> loadDat(const char* fileName) {
		std::string f;
		std::ifstream file(f.append(IMG_PATH).append(fileName).append(".dat"));
		std::string line;
		// skip first two lines
		std::getline(file, line);
		std::getline(file, line);
		std::vector<int> digits(81);
		int i = 0;
		while (std::getline(file, line) && i < 81) {
			for (char &b : line) {
				if (isdigit(b)) {
					digits[i++] = b - '0';
				}
			}
		}
		return digits;
	}

	static std::vector<std::string> findAllImages() {
		std::vector<std::string> fileNames;
		std::string f;
		std::ifstream file(TEST_FILE_LIST_PATH);
		std::string line;
		while (std::getline(file, line)) {
			fileNames.push_back(line);
		}
		return fileNames;
	}

	static std::vector<int> evalDigits(std::vector<int> &result, std::vector<int> &expected) {
        int total = 0;
        int found = 0;
        int correct = 0;
        int missed = 0;
        int wrong = 0;

        for (int i : expected) {
            if (i != 0)
                total++;
        }
        for (int i : result) {
            if (i != 0)
                found++;
        }
        missed = total - found;

        for (int i = 0; i < 9 * 9; i++) {
            if (expected[i] != 0 && result[i] != expected[i]) {
                wrong++;
            }
        }
        wrong += missed;
        correct = total - missed - wrong;

        return std::vector<int>{ total, found, correct, missed, wrong };
    }

	static void evalAll() {

		std::vector<std::string> imgs = findAllImages();

        int evals[5] = {0};
        int noGrid = 0;
        for (std::string img : imgs) {
        	std::vector<int> expected = loadDat(img.c_str());
        	std::string path;
			Mat imgMat = imread(path.append(IMG_PATH).append(img).append(".jpg"));
			std::string output;
			Detector detector = Detector(imgMat);
			if (!detector.detectPuzzle()){

				std::vector<int> eval = evalDigits(detector.extractedDigits, expected);
				for (unsigned int i = 0; i < eval.size(); i++) {
					evals[i] += eval[i];
				}
				Puzzle puzzle(detector.extractedDigits.data());
				if(!puzzle.solve()){

					Mat drawn = detector.drawDigits(puzzle.getResolvedDigits());

//					imwrite(output.append("../../resource/output/drawn/").append(img).append(".jpg"), drawn);

				} else {

					Mat drawn = detector.drawDigits(puzzle.getResolvedDigits());

//					imwrite(output.append("../../resource/output/drawn/Grid_Found_Not_Solved_").append(img).append(".jpg"), drawn);

				}


			} else {
//				imwrite(output.append("../../resource/output/drawn/Grid_Not_Found_").append(img).append(".jpg"), imgMat);

				noGrid++;
//				cout << "grid not found for " << img << "\n";
			}

        }
        std::cout << "grid not found : " << noGrid << "\n";
        std::cout << "Total: " << evals[0] <<
    			", found: " << evals[1] <<
				", correct: " << evals[2] <<
				", missed: " << evals[3] <<
				", wrong: " << evals[4] << "\n";
    }

	static void evalOne(const char* name) {
		// call cvtColor once to avoid incorrect timing
		Mat a = Mat(Size(5, 5), CV_8UC3);
    	cvtColor(a, a, COLOR_BGR2GRAY);

		Timer timer("eval one", true);
		std::vector<int> expected = loadDat(name);
		std::string path;
        Mat imgMat = imread(path.append(IMG_PATH).append(name).append(".jpg"));
        timer.timeit("read file");
		Detector detector = Detector(imgMat);
		timer.timeit("initialize detector");
		if (!detector.detectPuzzle()){
			timer.timeit("detecting");
			std::vector<int> eval = evalDigits(detector.extractedDigits, expected);
			timer.timeit("eval digits");
			Puzzle puzzle(detector.extractedDigits.data());
			timer.timeit("initialize solver");
			if(!puzzle.solve()){
				timer.timeit("solving");
				Mat drawn = detector.drawDigits(puzzle.getResolvedDigits());
				timer.timeit("draw digits");
				std::string output;
//				imwrite(output.append("../../resource/").append(name).append("Drawn.jpg"), drawn);
			}
			std::cout << "Total: " << eval[0] <<
        			", found: " << eval[1] <<
					", correct: " << eval[2] <<
					", missed: " << eval[3] <<
					", wrong: " << eval[4]<< "\n";
        }
		timer.timeall("all");
    }


private:
};
