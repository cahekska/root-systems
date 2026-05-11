#include <iostream>
#include <vector>
#include <bitset>
#include <set>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <queue>
#include <array>
#include <string>

using namespace std;

#ifndef L
#define L 5
#endif

const double EPS = 1e-8;
const double TOLERANCE = 1e-8;


const int MAX_VECTORS = 65536;

using Coverage = bitset<MAX_VECTORS>;

template <int N>
struct Vec {
	array<double, N> c{};

	Vec() { c.fill(0.0); }

	double dot(const Vec& other) const {
		double s = 0.0;
		for (int i = 0; i < N; i++) s += c[i] * other.c[i];
		return s;
	}

	Vec operator-() const {
		Vec r;
		for (int i = 0; i < N; i++) r.c[i] = -c[i];
		return r;
	}

	double norm() const { return sqrt(dot(*this)); }
};

template <int N>
struct Mat {
	array<array<double, N>, N> m{};

	Mat() {
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				m[i][j] = (i == j) ? 1.0 : 0.0;
	}

	Vec<N> operator*(const Vec<N>& v) const {
		Vec<N> r;
		for (int i = 0; i < N; i++) {
			double s = 0.0;
			for (int j = 0; j < N; j++) s += m[i][j] * v.c[j];
			r.c[i] = s;
		}
		return r;
	}

	Mat operator*(const Mat& other) const {
		Mat result;
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++) {
				double s = 0.0;
				for (int k = 0; k < N; k++) s += m[i][k] * other.m[k][j];
				result.m[i][j] = s;
			}
		return result;
	}
};

vector<Vec<L>> load_vectors(const string& filename) {
	vector<Vec<L>> vectors;
	ifstream file(filename);
	if (!file.is_open()) {
		cerr << "Error: cannot open " << filename << endl;
		return vectors;
	}
	string line;

	while (getline(file, line)) {
		istringstream iss(line);
		vector<double> coords;
		double val;
		while (iss >> val) coords.push_back(val);
		if ((int)coords.size() == L) {
			Vec<L> v;
			for (int i = 0; i < L; i++) v.c[i] = coords[i];
			vectors.push_back(v);
		}
	}

	cout << "Loaded " << vectors.size() << " vectors from " << filename << endl;
	return vectors;
}


vector<Vec<L>> generate_Dl_roots() {
	vector<Vec<L>> roots;
	for (int i = 0; i < L; i++) {
		for (int j = i + 1; j < L; j++) {
			for (int s1 : {1, -1}) {
				for (int s2 : {1, -1}) {
					Vec<L> v;
					v.c[i] = (double)s1;
					v.c[j] = (double)s2;
					roots.push_back(v);
				}
			}
		}
	}
	return roots;
}


Mat<L> reflection_matrix(const Vec<L>& alpha) {
	Mat<L> refl;
	double n2 = alpha.dot(alpha);
	for (int i = 0; i < L; i++)
		for (int j = 0; j < L; j++)
			refl.m[i][j] = (i == j ? 1.0 : 0.0) - 2.0 * alpha.c[i] * alpha.c[j] / n2;
	return refl;
}


vector<Mat<L>> generate_weyl_group(const vector<Vec<L>>& roots) {
	vector<Mat<L>> group;
	queue<Mat<L>> q;
	Mat<L> id;
	q.push(id);
	group.push_back(id);

	auto flatten = [](const Mat<L>& mat) {
		vector<double> flat(L * L);
		for (int i = 0; i < L; i++)
			for (int j = 0; j < L; j++)
				flat[i * L + j] = round(mat.m[i][j] * 1e8) / 1e8;
		return flat;
	};

	set<vector<double>> seen;
	seen.insert(flatten(id));

	while (!q.empty()) {
		Mat<L> g = q.front(); q.pop();
		for (const auto& root : roots) {
			Mat<L> ng = reflection_matrix(root) * g;
			auto f = flatten(ng);
			if (seen.find(f) == seen.end()) {
				seen.insert(f);
				group.push_back(ng);
				q.push(ng);
			}
		}
	}

	return group;
}


vector<Vec<L>> get_dominant_chamber() {
	static_assert(L >= 2, "D_l определена для l >= 2 (для l=2,3 — частные случаи)");
	vector<Vec<L>> dom;

	// Префикс-суммы e1+...+e_k для k = 1..l-2
	Vec<L> running;
	for (int k = 0; k < L - 2; k++) {
		running.c[k] = 1.0;
		dom.push_back(running);
	}

	// Предпоследний: e1+...+e_{l-1} - e_l
	Vec<L> v_minus = running;
	v_minus.c[L - 2] = 1.0;
	v_minus.c[L - 1] = -1.0;
	dom.push_back(v_minus);

	// Последний: e1+...+e_{l-1} + e_l
	Vec<L> v_plus = running;
	v_plus.c[L - 2] = 1.0;
	v_plus.c[L - 1] = 1.0;
	dom.push_back(v_plus);

	return dom;
}

vector<vector<Vec<L>>> generate_chambers() {
	cout << "Generating D" << L << " roots..." << endl;
	auto roots = generate_Dl_roots();
	cout << "Roots: " << roots.size()
	     << "  (expected " << 2 * L * (L - 1) << ")" << endl;

	cout << "Generating Weyl group..." << endl;
	auto group = generate_weyl_group(roots);
	cout << "Group order: " << group.size();
	long long expected = 1;
	for (int k = 1; k <= L; k++) expected *= k;
	expected *= (1LL << (L - 1));
	cout << "  (expected 2^(l-1)·l! = " << expected << ")" << endl;

	auto dominant = get_dominant_chamber();

	vector<vector<Vec<L>>> result;
	result.reserve(group.size());
	for (const auto& g : group) {
		vector<Vec<L>> chamber;
		chamber.reserve(L);
		for (const auto& v : dominant) chamber.push_back(g * v);
		result.push_back(std::move(chamber));
	}

	return result;
}

bool chamber_covers(const vector<Vec<L>>& chamber, const Vec<L>& v) {
	for (const auto& wall : chamber)
		if (wall.dot(v) <= TOLERANCE) return false;
	return true;
}

void save_bitsets(const vector<Coverage>& coverage, const string& filename, int n_vectors) {
	ofstream file(filename);
	for (size_t i = 0; i < coverage.size(); i++) {
		for (int j = 0; j < n_vectors; j++)
			file << (coverage[i][j] ? '1' : '0');
		file << '\n';
	}
	file.close();
	cout << "Saved " << coverage.size() << " bitsets to " << filename << endl;
}

void save_chambers(const vector<vector<Vec<L>>>& chambers, const string& filename) {
	ofstream file(filename);
	if (!file.is_open()) {
		cerr << "Error: Cannot open file " << filename << endl;
		return;
	}

	for (size_t i = 0; i < chambers.size(); i++) {
		file << "Chamber " << i << ":\n";
		for (const auto& v : chambers[i]) {
			for (int k = 0; k < L; k++) {
				file << v.c[k];
				if (k + 1 < L) file << " ";
			}
			file << "\n";
		}
		file << "\n";
	}

	file.close();
	cout << "Saved " << chambers.size() << " chambers to " << filename << endl;
}

vector<vector<Vec<L>>> load_chambers(const string& filename) {
	vector<vector<Vec<L>>> chambers;
	ifstream file(filename);
	string line;

	vector<Vec<L>> current_chamber;

	while (getline(file, line)) {
		if (line.empty()) continue;

		if (line.find("Chamber ") == 0) {
			if (!current_chamber.empty()) {
				chambers.push_back(current_chamber);
				current_chamber.clear();
			}
			continue;
		}

		istringstream iss(line);
		Vec<L> v;
		bool ok = true;
		for (int k = 0; k < L; k++) {
			if (!(iss >> v.c[k])) { ok = false; break; }
		}
		if (ok) current_chamber.push_back(v);
	}

	if (!current_chamber.empty()) chambers.push_back(current_chamber);

	cout << "Loaded " << chambers.size() << " chambers from " << filename << endl;
	return chambers;
}

int main(int argc, char** argv) {
	ios::sync_with_stdio(false);

	string vectors_file   = "d" + to_string(L) + "_characteristic_set.txt";
	string chambers_file  = "chambers_d" + to_string(L) + ".txt";
	string coverage_file  = "coverage_bitsets_d" + to_string(L) + ".txt";

	if (argc >= 2) vectors_file  = argv[1];
	if (argc >= 3) chambers_file = argv[2];
	if (argc >= 4) coverage_file = argv[3];

	cout << "=== D" << L << " ===" << endl;
	cout << "Vectors:  " << vectors_file  << endl;
	cout << "Chambers: " << chambers_file << endl;
	cout << "Coverage: " << coverage_file << endl;

	auto vectors = load_vectors(vectors_file);
	int n = (int)vectors.size();
	cout << "Vector count: " << n << endl;

	if (n == 0) {
		cerr << "No vectors loaded — aborting." << endl;
		return 1;
	}
	if (n > MAX_VECTORS) {
		cerr << "Too many vectors (" << n << ") — increase MAX_VECTORS (" 
		     << MAX_VECTORS << ")" << endl;
		return 1;
	}

	auto chambers = generate_chambers();
	cout << "Chamber count: " << chambers.size() << endl;

	save_chambers(chambers, chambers_file);

	cout << "Computing coverage..." << endl;
	vector<Coverage> coverage(chambers.size());

	for (size_t i = 0; i < chambers.size(); i++) {
		for (int j = 0; j < n; j++)
			if (chamber_covers(chambers[i], vectors[j]))
				coverage[i].set(j);

		if ((i + 1) % 100 == 0)
			cout << "  " << (i + 1) << "/" << chambers.size()
			     << " (" << coverage[i].count() << " covered)" << endl;
	}

	save_bitsets(coverage, coverage_file, n);

	cout << "Done!" << endl;
	return 0;
}
