/*
 * Train.cpp
 *
 *  Created on: 20 sept. 2017
 *      Author: E507067
 */

#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "Detector.h"


using namespace ml;

#define DIGITS_PATH "../../resource/28x28/"

#define NB_IMG 140

class Train {
public:
	Train() {

	}

	~Train() {

	}

	static void trainSvm() {

        Ptr<ml::SVM> newSvm = SVM::create();
        newSvm->setKernel(SVM::LINEAR);
        newSvm->setType(SVM::C_SVC);
        newSvm->setC(2.67);
        newSvm->setGamma(5.383);

		std::vector<int> resp(SZ * NB_IMG);
		for (int i = 0; i < SZ * NB_IMG; i++){
			resp[i] = i / NB_IMG + 1;
		}

		std::vector<float> samples;
		for (int i = 1; i < 10; i++){
			std::vector<std::string> fileNames;
			std::string f;

			std::ifstream imgListFile(f.append(DIGITS_PATH).append(std::to_string(i)).append(".txt"));
			std::string line;
			while (getline(imgListFile, line)) {
				fileNames.push_back(line);
			}
			imgListFile.close(); // for clarity only

			for (std::string fileName : fileNames){

				std::string im;
				Mat img = imread(im.append(DIGITS_PATH).append(std::to_string(i)).append("/").append(fileName));

				cvtColor(img, img, COLOR_BGR2GRAY);
				Detector::deskew(img, img);
				std::vector<float> hogData(NB_AREA * NB_BIN);
				Detector::hog(img, hogData);

				for(float hog : hogData){
					samples.push_back(hog);
				}
			}
		}

		Mat testData = Mat(SZ * NB_IMG, NB_AREA * NB_BIN, CV_32F, samples.data());
		Mat respData = Mat(SZ * NB_IMG, 1, CV_32S, resp.data());

		newSvm->train(testData , ml::ROW_SAMPLE , respData);
		newSvm->save(SVM_PATH);

	}

	static void trainAutoSvm() {

	        Ptr<ml::SVM> newSvm = SVM::create();
	        newSvm->setKernel(SVM::LINEAR);
	        newSvm->setType(SVM::C_SVC);
	        newSvm->setC(2.67);
	        newSvm->setGamma(5.383);

			std::vector<int> resp(SZ * NB_IMG);
			for (int i = 0; i < SZ * NB_IMG; i++){
				resp[i] = i / NB_IMG + 1;
			}

			std::vector<float> samples;
			for (int i = 1; i < 10; i++){
				std::vector<std::string> fileNames;
				std::string f;

				std::ifstream imgListFile(f.append(DIGITS_PATH).append(std::to_string(i)).append(".txt"));
				std::string line;
				while (std::getline(imgListFile, line)) {
					fileNames.push_back(line);
				}
				imgListFile.close(); // for clarity only

				for (std::string fileName : fileNames){

					std::string im;
					Mat img = imread(im.append(DIGITS_PATH).append(std::to_string(i)).append("/").append(fileName));

					cvtColor(img, img, COLOR_BGR2GRAY);
					// Optional
					// Detector::deskew(img, img);

					std::vector<float> hogData(NB_AREA * NB_BIN);
					Detector::hog(img, hogData);

					for(float hog : hogData){
						samples.push_back(hog);
					}
				}
			}

			Mat testData = Mat(SZ * NB_IMG, NB_AREA * NB_BIN, CV_32F, samples.data());
			Mat respData = Mat(SZ * NB_IMG, 1, CV_32S, resp.data());

			newSvm->train(testData , ml::ROW_SAMPLE , respData);
			newSvm->save(SVM_PATH);

		}
private:
};
