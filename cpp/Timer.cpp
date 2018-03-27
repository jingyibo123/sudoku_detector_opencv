/*
 * Timeit.cpp
 *
 *  Created on: 20 sept. 2017
 *      Author: E507067
 */

#ifndef SRC_TIMER_CPP_
#define SRC_TIMER_CPP_
#include <opencv2/core/core.hpp>

#include <iostream>

class Timer{
public:
	Timer(char const* tag, bool debug = false){
		this->tag = tag;
		this->debug = debug;
		this->origin = cvGetTickCount();
		this->current = cvGetTickCount();
	}

	~Timer(){

	}

	void timeit(char const* msg = ""){
		if(debug && strlen(msg) > 0 ){
			printf("__________%s : Time taken: %.2f ms \t  %s \n", tag,
					(cvGetTickCount() - current) / (cvGetTickFrequency()*1000.0), msg);
		}
		current = cvGetTickCount();
	}

	double timeitDouble(){
		double old = current;
		current = cvGetTickCount();
		return (cvGetTickCount() - old) / (cvGetTickFrequency()*1000.0);
	}

	void timeall(char const* msg){
		if(debug){
			printf("__________%s : Overall time taken: %.2f ms  %s \n", tag,
					(cvGetTickCount() - origin) / (cvGetTickFrequency()*1000.0), msg);
		}
		current = cvGetTickCount();
	}
private:
	char const* tag;
	bool debug;
	double origin;
	double current;


};
#endif /* SRC_TIMER_CPP_ */
