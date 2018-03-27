#include "Puzzle.h"

bool Puzzle::initiated;

int Puzzle::peers[SZ][SZ][NB_PEERS];

int Puzzle::rowPeers[SZ][SZ][SZ - 1];

int Puzzle::colPeers[SZ][SZ][SZ - 1];

int Puzzle::blockPeers[SZ][SZ][SZ - 1];

int (*Puzzle::allPeersGroup[3])[9][8];

void Puzzle::init() {
	if (initiated)
		return;
	// Initialize peers
	for (int i = 0; i < SZ; i++) {
		for (int j = 0; j < SZ; j++) {
			int k = 0;
			// column
			for (int r = 0; r < SZ; r++) {
				if (r == i)
					continue;
				peers[i][j][k++] = (r * SZ + j);
			}
			// row
			for (int c = 0; c < SZ; c++) {
				if (c == j)
					continue;
				peers[i][j][k++] = (i * SZ + c);
			}
			// block
			for (int n = 0; n < SZ * SZ; n++) {
				if (n / SZ == i || n % SZ == j || (n / SZ == i && n % SZ == j))
					continue;
				if (i / 3 == n / 27 && j / 3 == n % 9 / 3)
					peers[i][j][k++] = n;
			}
		}
	}

	// Initialize rowPeers
	for (int i = 0; i < SZ; i++) {
		for (int j = 0; j < SZ; j++) {
			int k = 0;
			// row
			for (int c = 0; c < SZ; c++) {
				if (c == j)
					continue;
				rowPeers[i][j][k++] = (i * SZ + c);
			}
		}
	}

	// Initialize colPeers
	for (int i = 0; i < SZ; i++) {
		for (int j = 0; j < SZ; j++) {
			int k = 0;
			// column
			for (int r = 0; r < SZ; r++) {
				if (r == i)
					continue;
				colPeers[i][j][k++] = (r * SZ + j);
			}
		}
	}

	// Initialize blockPeers
	for (int i = 0; i < SZ; i++) {
		for (int j = 0; j < SZ; j++) {
			int k = 0;
			// block
			for (int n = 0; n < SZ * SZ; n++) {
				if (i / 3 == n / 27 && j / 3 == n % 9 / 3 && n != i * 9 + j)
					blockPeers[i][j][k++] = n;
			}
		}
	}

	allPeersGroup[0] = rowPeers;
	allPeersGroup[1] = colPeers;
	allPeersGroup[2] = blockPeers;
	initiated = true;
}

Puzzle::Puzzle(const char* s) {
	Puzzle::init();

	// Initialize options
	memset(options, true, sizeof(bool) * SZ * SZ * SZ);

	// copy the initial values
	for (int i = 0; i < SZ * SZ; i++) {
		if (s[i] >= '0' && s[i] <= '9') {
			initValues[i] = s[i] - '0';
		} else {
			initValues[i] = 0;
		}
	}
}

Puzzle::Puzzle(int s[]) {
	Puzzle::init();

	// Initialize options
	memset(options, true, sizeof(bool) * SZ * SZ * SZ);

	// copy the initial values
	memcpy(initValues, s, sizeof(initValues));
}

/**
Use this copy constructor to copy values and options to create search branches
 */
Puzzle::Puzzle(const Puzzle &source) {

	nbAssignedValues = source.nbAssignedValues;
	memset(initValues, 0, sizeof(int) * SZ * SZ);
	memcpy(options, source.options, sizeof (bool) * SZ * SZ * SZ);
	memcpy(values, source.values, sizeof (int) * SZ * SZ);
}

Puzzle::~Puzzle() {

}


bool Puzzle::validateInput() {
	int nbKnown = 0;
	int digits[SZ] = {0};
	for (int i = 0; i < SZ * SZ; i++) {
		if(initValues[i] != 0){
			digits[initValues[i] - 1] = 1;
			nbKnown++;
		}
	}
	// Must have at lest 17 cells known
	if(nbKnown < 17)
		return false;

	int nbDigits = digits[0] + digits[1] + digits[2] + digits[3] + digits[4] + digits[5] + digits[6] + digits[7] + digits[8];
	// Must have at least 8 numbers known
	if(nbDigits < 8)
		return false;

	return true;
}
/**
Assign all values in initValues which is: values in initial input for the root branch, or the asserted value for a search branch.

@return false when contradiction occurred
 */
bool Puzzle::assignAll() {
	for (int i = 0; i < SZ * SZ; i++) {
		if (initValues[i] > 0)
			if (!assignOne(i / SZ, i % SZ, initValues[i]))
				return false;

	}
	return true;
}

/**
Assign a value to an unknown cell and propagate the elimination.

@param row the row id 0-8
@param col the column id 0-8
@param val the new value 1-9
@return false when contradiction occurred
 */
bool Puzzle::assignOne(int row, int col, int val) {

	// get all other possible options other than val
	int nbOps = 0;
	int ops[SZ];
	for (int i = 0; i < SZ; i++) {
		if (options[row][col][i] && i != val - 1)
			ops[nbOps++] = (i + 1);
	}

	// eliminate all the other values
	for (int i = 0; i < nbOps; i++) {
		if (!eliminate(row, col, ops[i]))
			return false;
	}
	return true;
}

/**
Eliminate an option from a cell and analyze the impact.

@param row the row id 0-8
@param col the column id 0-8
@param val the new value 1-9
@return false when contradiction occurred
 */
bool Puzzle::eliminate(int row, int col, int val) {
	if (!options[row][col][val - 1]) {
		return true;
	}
	// eliminate that option
	options[row][col][val - 1] = false;
	// get all the options left
	int nbOps = 0;
	int ops[SZ];
	for (int i = 0; i < SZ; i++) {
		if (options[row][col][i]) {
			ops[nbOps++] = (i + 1);
		}
	}

	// if none left after elimination, return error
	if (nbOps == 0)
		return false;
	// if only one left and that cell is not known, assign it with that value
	if (nbOps == 1) {
		int op = ops[0];
		values[row][col] = op;
		nbAssignedValues++;
		// eliminate val on all its peers
		for (int peer : peers[row][col]) {
			if (!eliminate(peer / SZ, peer % SZ, op))
				return false;
		}
	}
	// See if after eliminating val, its peers(row, column or block) contain only one place for val
	for (int (*somePeers)[9][8]  : allPeersGroup) {

		int nbPos = 0;
		int pos[SZ];
		// Get all peer cells where val is possible
		for (int peerPos : somePeers[row][col]) {
			if (options[peerPos / SZ][peerPos % SZ][val - 1]) {
				pos[nbPos++] = peerPos;
			}
		}

		if (nbPos == 0)
		return false;

		if (nbPos == 1) {
			int p = pos[0];
			return assignOne(p / SZ, p % SZ, val);
		}
	}

	return true;
}

/**
Attempt to solve the puzzle with conventional algorithm (assign() and eliminate()) and start a recursive search if still unsolved

@return false when contradiction occurred
 */
int Puzzle::solve() {
	// only validate input for the root branch
	if(nbAssignedValues == 0 && !validateInput())
		return PUZZLE_NOT_ENOUGH_DIGITS;

	if (!assignAll())
		return PUZZLE_NO_SOLUTION;
	if (nbAssignedValues == SZ * SZ){
		solved = true;
		return PUZZLE_SOLVED;
//		return validate();
	}

	// if there's still uncertain cell

	// find number of options for all cells
	int nbOptions[SZ][SZ] = {};
	for (int i = 0; i < SZ; i++) {
		for (int j = 0; j < SZ; j++) {
			for (int k = 0; k < SZ; k++) {
				nbOptions[i][j] += options[i][j][k] ? 1 : 0;
			}
		}
	}
	// find the position with the fewest options
	int nbOps = SZ;
	int pos = 0; // 0-80
	for (int i = 0; i < SZ; i++) {
		for (int j = 0; j < SZ; j++) {
			if (nbOptions[i][j] > 1 && nbOptions[i][j] < nbOps) {
				nbOps = nbOptions[i][j];
				pos = (i * SZ + j);
			}
			if (nbOps == 2)
				break;
		}
	}
	// get all options of that position
	int nb = 0;
	int ops[SZ]; // 1 - 9
	for (int i = 0; i < SZ; i++) {
		if (options[pos / SZ][pos % SZ][i])
			ops[nb++] = (i + 1);
	}
	// try each option with a new object
	for (int l = 0; l < nb; l++) {
		Puzzle branch(*this);
		branch.initValues[pos] = ops[l];

		// if solved
		if (!branch.solve()) {
			// copy back all values
			nbAssignedValues = branch.nbAssignedValues;
//			memcpy(options, branch.options, sizeof (bool) * SZ * SZ * SZ);
			memcpy(values, branch.values, sizeof (int) * SZ * SZ);
			solved = true;
			return PUZZLE_SOLVED;
		}
	}
	// still no solution after searching
	return PUZZLE_NO_SOLUTION;
}

/**
Test result by summing all peers

@return false when contradiction occurred
 */
bool Puzzle::validate() {

	// First make sure we have correct number of all digits
	int count[SZ] = {};
	for (int i = 0; i < SZ * SZ; i++) {
		count[values[i / SZ][i % SZ] - 1]++;
	}
	for (int c : count) {
		if (c != SZ)
			return false;
	}

	for (int i = 0; i < SZ; i++) {
		// lines
		int sum = 0;
		for (int j = 0; j < SZ; j++) {
			sum += values[i][j];
		}
		if (sum != 45)
			return false;
		// columns
		sum = 0;
		for (int j = 0; j < SZ; j++) {
			sum += values[j][i];
		}
		if (sum != 45)
			return false;
	}
	// blocks
	int blocks[SZ] = {0, 3, 6, 27, 30, 33, 54, 57, 60};
	for (int i : blocks) {
		int sum = values[i / SZ][i % SZ];
		for (int j : blockPeers[i / SZ][i % SZ]) {
			sum += values[j / SZ][j % SZ];
		}
		if (sum != 45)
			return false;

	}
	solved = true;
	return true;
}

std::vector<int> Puzzle::getResolvedDigits(){

	std::vector<int> digits(SZ*SZ);
	memcpy((void*)digits.data(), this->values, sizeof (int) * SZ * SZ);
	return digits;
}
/*
void Puzzle::disp(){
	string sb;
	for (int i = 0; i < SZ; i++) {
		for (int j = 0; j < SZ; j++) {
			sb.append(to_string(values[i][j])).append(" ");
			if ((j + 1) % 3 == 0 && j != 8)
				sb.append("|");
		}
		sb.append("\n");
		if ((i + 1) % 3 == 0 && i != 8)
			sb.append("------+------+------\n");
	}

	cout << sb;
}
*/
