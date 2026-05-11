#include <bits/stdc++.h>
using namespace std;

static const int ROWS  = 1152;
static const int BITS  = 37776; // количество векторов в хар наборе
static const int WORDS = (BITS + 63) / 64;

// Хранение строк в упакованном виде
uint64_t rows[ROWS][WORDS];

// suffixOr[i][w] = OR-сумма rows[i..ROWS-1] по слову w
uint64_t suffixOr[ROWS + 1][WORDS];

// bitRows[b] — отсортированный список строк, в которых бит b равен 1
vector<int> bitRows[BITS];

int solution[6];   // ответ

inline bool needIsZero(const uint64_t need[]) {
	for (int i = 0; i < WORDS; i++)
		if (need[i]) return false;
	return true;
}

// Рекурсивный поиск с отсечениями
//
// depth    — текущая глубина (0 = только строка 0 зафиксирована)
// maxDepth — максимальная глубина (5 — нужно ещё 5 строк)
// need[]   — биты, которые ещё не покрыты
// startIdx — минимальный допустимый индекс следующей строки
bool solve(int depth, int maxDepth,
		   const uint64_t need[], int startIdx)
{
	if (needIsZero(need))
		return true;
	if (depth == maxDepth)
		return false;
	
	// Отсечение 1: suffixOr начиная с startIdx должен покрыть все нужные биты
	for (int i = 0; i < WORDS; i++)
		if (need[i] & ~suffixOr[startIdx][i])
			return false;
	
	// Выбираем «наиболее ограниченный» бит из need:
	// тот, который покрывается наименьшим числом строк с индексом >= startIdx.
	int bestBit   = -1;
	int bestCount = INT_MAX;
	
	for (int w = 0; w < WORDS && bestCount > 1; w++) {
		uint64_t word = need[w];
		while (word && bestCount > 1) {
			int pos = __builtin_ctzll(word);
			word &= word - 1;
			int b = w * 64 + pos;
			
			// Кол-во строк >= startIdx, покрывающих бит b
			const auto& br = bitRows[b];
			int cnt = (int)(br.end()
							- lower_bound(br.begin(), br.end(), startIdx));
			if (cnt < bestCount) {
				bestCount = cnt;
				bestBit   = b;
			}
		}
	}
	
	if (bestCount == 0 || bestBit == -1)
		return false;   // бит не может быть покрыт - тупик
	
	// Ветвимся по строкам >= startIdx, покрывающим bestBit
	const auto& br = bitRows[bestBit];
	auto it = lower_bound(br.begin(), br.end(), startIdx);
	
	for (; it != br.end(); ++it) {
		int idx = *it;
		solution[depth + 1] = idx;
		
		// newNeed = need & ~rows[idx]
		uint64_t newNeed[WORDS];
		for (int i = 0; i < WORDS; i++)
			newNeed[i] = need[i] & ~rows[idx][i];
		
		if (solve(depth + 1, maxDepth, newNeed, idx + 1))
			return true;
	}
	return false;
}


int main(int argc, char* argv[])
{
	ios_base::sync_with_stdio(false);
	cin.tie(nullptr);
	
	const char* filename = "coverage_bitsetsf_least.txt";
	
	cerr << "Reading " << filename << " ...\n";
	ifstream fin(filename);
	if (!fin) {
		cerr << "Error: cannot open '" << filename << "'\n";
		return 1;
	}
	
	memset(rows, 0, sizeof(rows));
	
	for (int r = 0; r < ROWS; r++) {
		string line;
		fin >> line;
		if (!fin) {
			cerr << "Error: only " << r << " rows in file (expected " << ROWS << ")\n";
			return 1;
		}
		if ((int)line.size() != BITS) {
			cerr << "Error: row " << r << " has length " << line.size()
			<< " (expected " << BITS << ")\n";
			return 1;
		}
		for (int b = 0; b < BITS; b++)
			if (line[b] == '1')
				rows[r][b / 64] |= (1ULL << (b % 64));
	}
	fin.close();
	cerr << "Done reading.\n";
	
	cerr << "Building bitRows ...\n";
	for (int r = 0; r < ROWS; r++)
		for (int b = 0; b < BITS; b++)
			if ((rows[r][b / 64] >> (b % 64)) & 1)
				bitRows[b].push_back(r);   // r возрастает → уже отсортировано
	
	cerr << "Building suffixOr ...\n";
	memset(suffixOr[ROWS], 0, sizeof(suffixOr[ROWS]));
	for (int r = ROWS - 1; r >= 0; r--)
		for (int i = 0; i < WORDS; i++)
			suffixOr[r][i] = suffixOr[r + 1][i] | rows[r][i];
	
	for (int i = 0; i < WORDS - 1; i++) {
		if (suffixOr[0][i] != UINT64_MAX) {
			cerr << "Warning: union of all rows does not cover all bits! "
			<< "(word " << i << ")\n";
		}
	}
	{
		int rem = BITS % 64;
		uint64_t mask = (rem == 0) ? UINT64_MAX : ((1ULL << rem) - 1);
		if ((suffixOr[0][WORDS - 1] & mask) != mask)
			cerr << "Warning: union of all rows does not cover all bits! (last word)\n";
	}
	
	solution[0] = 0;
	uint64_t need[WORDS];
	for (int i = 0; i < WORDS; i++)
		need[i] = ~rows[0][i];
	// Обнуляем «лишние» биты в последнем слове
	if (BITS % 64 != 0)
		need[WORDS - 1] &= (1ULL << (BITS % 64)) - 1;

	cerr << "Searching ...\n";
	auto t0 = chrono::steady_clock::now();
	
	bool found = solve(0, 5, need, 1);   // сколько еще строк(камер) кроме первой найти
	
	auto t1 = chrono::steady_clock::now();
	double sec = chrono::duration<double>(t1 - t0).count();
	
	if (found) {
		cout << "Solution found in " << sec << " s\n";
		cout << "Row indices (1-based):\n";
		for (int i = 0; i < 6; i++)
			cout << "  " << (solution[i] + 1) << "\n";		
		
		uint64_t orAll[WORDS] = {};
		for (int i = 0; i < 6; i++) 
			for (int w = 0; w < WORDS; w++)
				orAll[w] |= rows[solution[i]][w];
		
		bool ok = true;
		for (int w = 0; w < WORDS - 1; w++)
			if (orAll[w] != UINT64_MAX) { ok = false; break; }
		if (ok) {
			int rem = BITS % 64;
			uint64_t mask = (rem == 0) ? UINT64_MAX : ((1ULL << rem) - 1);
			if ((orAll[WORDS - 1] & mask) != mask) ok = false;
		}
		cout << (ok ? "Verification: OK\n" : "Verification: FAILED!\n");
	} else {
		cout << "No solution found (" << sec << " s)\n";
	}
	
	return 0;
}
