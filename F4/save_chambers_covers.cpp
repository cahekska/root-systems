#include <iostream>
#include <vector>
#include <bitset>
#include <set>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <queue>

using namespace std;

const double EPS = 1e-8;
const double TOLERANCE = 1e-8;
const int MAX_VECTORS = 40000;

using Coverage = bitset<MAX_VECTORS>;

struct Vec4 {
	double x, y, z, w;
	
	Vec4() : x(0), y(0), z(0), w(0) {}
	Vec4(double x, double y, double z, double w) : x(x), y(y), z(z), w(w) {}
	
	double dot(const Vec4& other) const {
		return x * other.x + y * other.y + z * other.z + w * other.w;
	}
	
	Vec4 operator-() const {
		return Vec4(-x, -y, -z, -w);
	}
	
	Vec4 operator-(const Vec4& other) const {
		return Vec4(x - other.x, y - other.y, z - other.z, w - other.w);
	}
	
	Vec4 operator*(double s) const {
		return Vec4(x * s, y * s, z * s, w * s);
	}
	
	double norm() const {
		return sqrt(x*x + y*y + z*z + w*w);
	}
	
	Vec4 normalized() const {
		double n = norm();
		if (n > EPS) return Vec4(x/n, y/n, z/n, w/n);
		return *this;
	}
};

struct Mat4 {
	double m[4][4];
	
	Mat4() {
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				m[i][j] = (i == j) ? 1.0 : 0.0;
	}
	
	Vec4 operator*(const Vec4& v) const {
		return Vec4(
					m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z + m[0][3]*v.w,
					m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z + m[1][3]*v.w,
					m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z + m[2][3]*v.w,
					m[3][0]*v.x + m[3][1]*v.y + m[3][2]*v.z + m[3][3]*v.w
					);
	}
	
	Mat4 operator*(const Mat4& other) const {
		Mat4 result;
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++)
					result.m[i][j] += m[i][k] * other.m[k][j];
			}
		return result;
	}
};

vector<Vec4> load_vectors(const string& filename) {
	vector<Vec4> vectors;
	ifstream file(filename);
	string line;
	
	while (getline(file, line)) {
		istringstream iss(line);
		vector<double> coords;
		double val;
		while (iss >> val) coords.push_back(val);
		if (coords.size() == 4)
			vectors.push_back(Vec4(coords[0], coords[1], coords[2], coords[3]));
	}
	
	cout << "Loaded " << vectors.size() << " vectors from " << filename << endl;
	return vectors;
}

vector<Vec4> generate_F4_roots() {
	vector<Vec4> roots;
	
	// +-e_i
	for (int i = 0; i < 4; i++)
		for (int sign : {1, -1}) {
			double c[4] = {0,0,0,0};
			c[i] = sign;
			roots.push_back(Vec4(c[0], c[1], c[2], c[3]));
		}
	
	// +-e_i +- e_j
	for (int i = 0; i < 4; i++)
		for (int j = i+1; j < 4; j++)
			for (int s1 : {1, -1})
				for (int s2 : {1, -1}) {
					double c[4] = {0,0,0,0};
					c[i] = s1; c[j] = s2;
					roots.push_back(Vec4(c[0], c[1], c[2], c[3]));
				}
	
	// (+-1/2, +-1/2, +-1/2, +-1/2)
	for (int s1 : {1, -1})
		for (int s2 : {1, -1})
			for (int s3 : {1, -1})
				for (int s4 : {1, -1})
					roots.push_back(Vec4(s1*0.5, s2*0.5, s3*0.5, s4*0.5));
	
	// Уникальные ненулевые
	set<vector<double>> unique;
	for (const auto& r : roots) {
		if (r.norm() > EPS) {
			vector<double> rounded = {
				round(r.x * 1e8) / 1e8,
				round(r.y * 1e8) / 1e8,
				round(r.z * 1e8) / 1e8,
				round(r.w * 1e8) / 1e8
			};
			unique.insert(rounded);
		}
	}
	
	vector<Vec4> result;
	for (const auto& r : unique)
		result.push_back(Vec4(r[0], r[1], r[2], r[3]));
	
	return result;
}

Mat4 reflection_matrix(const Vec4& alpha) {
	Mat4 refl;
	double n2 = alpha.dot(alpha);
	double c[4] = {alpha.x, alpha.y, alpha.z, alpha.w};
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			refl.m[i][j] = (i == j ? 1.0 : 0.0) - 2.0 * c[i] * c[j] / n2;
	return refl;
}

vector<Mat4> generate_weyl_group(const vector<Vec4>& roots) {
	vector<Mat4> group;
	queue<Mat4> q;
	q.push(Mat4());
	group.push_back(Mat4());
	
	auto flatten = [](const Mat4& mat) {
		vector<double> flat(16);
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				flat[i*4+j] = round(mat.m[i][j] * 1e8) / 1e8;
		return flat;
	};
	
	set<vector<double>> seen;
	seen.insert(flatten(Mat4()));
	
	while (!q.empty()) {
		Mat4 g = q.front(); q.pop();
		for (const auto& root : roots) {
			Mat4 ng = reflection_matrix(root) * g;
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

vector<Vec4> get_dominant_chamber() {
	return {
	Vec4(1, 0, 0, 0),
	Vec4(1, 1, 0, 0),
	Vec4(2, 1, 1, 0),
	Vec4(1.5, 0.5, 0.5, 0.5)
};
}

vector<vector<Vec4>> generate_chambers() {
	cout << "Generating F4 roots..." << endl;
	auto roots = generate_F4_roots();
	cout << "Roots: " << roots.size() << endl;
	
	cout << "Generating Weyl group..." << endl;
	auto group = generate_weyl_group(roots);
	cout << "Group order: " << group.size() << endl;
	
	auto dominant = get_dominant_chamber();
	
	set<vector<vector<double>>> unique;
	vector<vector<Vec4>> result;
	
	for (const auto& g : group) {
		vector<Vec4> chamber;
		for (const auto& v : dominant)
			chamber.push_back(g * v);
		
		vector<vector<double>> rounded;
		for (const auto& v : chamber)
			rounded.push_back({
			round(v.x * 1e8) / 1e8,
			round(v.y * 1e8) / 1e8,
			round(v.z * 1e8) / 1e8,
			round(v.w * 1e8) / 1e8
		});
		sort(rounded.begin(), rounded.end());
		
		if (unique.find(rounded) == unique.end()) {
			unique.insert(rounded);
			result.push_back(chamber);
		}
	}
	
	return result;
}

bool chamber_covers(const vector<Vec4>& chamber, const Vec4& v) {
	for (const auto& wall : chamber)
		if (wall.dot(v) <= TOLERANCE)
			return false;
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

void save_chambers(const vector<vector<Vec4>>& chambers, const string& filename) {
	ofstream file(filename);
	if (!file.is_open()) {
		cerr << "Error: Cannot open file " << filename << endl;
		return;
	}
	
	for (size_t i = 0; i < chambers.size(); i++) {
		file << "Chamber " << i << ":\n";
		for (const auto& v : chambers[i]) {
			file << v.x << " " << v.y << " " << v.z << " " << v.w << "\n";
		}
		file << "\n";
	}
	
	file.close();
	cout << "Saved " << chambers.size() << " chambers to " << filename << endl;
}

vector<vector<Vec4>> load_chambers(const string& filename) {
	vector<vector<Vec4>> chambers;
	ifstream file(filename);
	string line;
	
	int chamber_idx = -1;
	vector<Vec4> current_chamber;
	
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
		double x, y, z, w;
		if (iss >> x >> y >> z >> w) {
			current_chamber.push_back(Vec4(x, y, z, w));
		}
	}
	
	if (!current_chamber.empty()) {
		chambers.push_back(current_chamber);
	}
	
	cout << "Loaded " << chambers.size() << " chambers from " << filename << endl;
	return chambers;
}

int main() {
	ios::sync_with_stdio(false);
	
	auto vectors = load_vectors("f4_least_popular.txt");
	int n = vectors.size();
	cout << "Vector count: " << n << endl;
	
	if (n > MAX_VECTORS) {
		cerr << "Too many vectors! Increase MAX_VECTORS" << endl;
		return 1;
	}
	
	auto chambers = generate_chambers();
	cout << "Chamber count: " << chambers.size() << endl;
	
	save_chambers(chambers, "chambers.txt");
	
	cout << "Computing coverage..." << endl;
	vector<Coverage> coverage(chambers.size());
	
	for (size_t i = 0; i < chambers.size(); i++) {
		for (int j = 0; j < n; j++)
			if (chamber_covers(chambers[i], vectors[j]))
				coverage[i].set(j);
		
		if ((i+1) % 100 == 0)
			cout << "  " << (i+1) << "/" << chambers.size() 
			<< " (" << coverage[i].count() << " covered)" << endl;
	}
	
	save_bitsets(coverage, "coverage_bitsets.txt", n);	
	cout << "Done!" << endl;
	return 0;
}
