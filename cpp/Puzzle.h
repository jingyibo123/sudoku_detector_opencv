/*
 * Detector.h
 *
 *  Created on: 19 sept. 2017
 *      Author: Hypos
 */

#ifndef SRC_PUZZLE_H_
#define SRC_PUZZLE_H_

#include <cstring>
#include <vector>

#define SZ 9

#define PUZZLE_SOLVED 0
#define PUZZLE_NOT_ENOUGH_DIGITS 1
#define PUZZLE_NO_SOLUTION 2

class Puzzle {
public:
	Puzzle(const char* s);
	Puzzle(int s[]);
	~Puzzle();

	bool solved = false;

	static void init();

	int solve();

	std::vector<int> getResolvedDigits();

//	void disp();

private:

	Puzzle(const Puzzle &source);

	bool validateInput();

	bool assignAll();

	bool assignOne(int row, int col, int val);

	bool eliminate(int row, int col, int val);

	bool validate();

	static bool initiated;

	// A cell always has 20 peers
	const static int NB_PEERS = 3 * (SZ - 1) - 4;

	// Array of the position in 0-80 form of all peers of every cell
	// For example, peers of last cell will be peers[8][8] = {8, 17... 72, 73, ...60, 61, 69, 70}
	static int peers[SZ][SZ][NB_PEERS];

    // Array of the position in 0-80 form of all row peers of every cell
    // For example, row peers of last cell will be peers[8][8] = {72, 73, 74, 75, 76, 77, 78, 79}
	static int rowPeers[SZ][SZ][SZ - 1];

	// Array of the position in 0-80 form of all column peers of every cell
	// For example, column peers of last cell will be peers[8][8] = {8, 17, 26, 35, 44, 53, 62, 71}
	static int colPeers[SZ][SZ][SZ - 1];

	// Array of the position in 0-80 form of all block peers of every cell
	// For example, block peers of last cell will be peers[8][8] = {60, 61, 62, 69, 70, 71, 78, 79}
	static int blockPeers[SZ][SZ][SZ - 1];

	// The array of all three types of peer group for iteration
	static int (*allPeersGroup[3])[9][8];


	// The matrix storing: all known values from the input string in case of the root branch, or the asserted value in case of a search branch
	int initValues[SZ * SZ];

	// The white board storing all values resolved
	int values[SZ][SZ] = {};

	// Count of values already assigned into matrix values
	int nbAssignedValues = 0;

	// Matrix of all possible options for each cell, options[row][col][val-1]
	// For example, options[2][4][6] == false --> 7 is eliminated for cell C5
	bool options[SZ][SZ][SZ];

};
#endif /* SRC_PUZZLE_H_ */
