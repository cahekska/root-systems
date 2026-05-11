#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <array>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>

#include <cuda_runtime.h>

#include <thread>
#include <mutex>

using i64  = long long;
using u64  = unsigned long long;
using i128 = __int128_t;
using u128 = __uint128_t;

static constexpr int  N  = 8;
static constexpr int  L  = 6;
static constexpr int  K  = L - 1;
static constexpr u64  PRIME = 2147483647ull;

#define CUDA_CHECK(x) do {                                              \
    cudaError_t err = (x);                                              \
    if (err != cudaSuccess){                                            \
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",            \
                     #x, __FILE__, __LINE__, cudaGetErrorString(err));  \
        std::exit(1);                                                   \
    }                                                                   \
} while(0)



static i64 gcd_i64(i64 a, i64 b){
    if (a<0) a=-a; if (b<0) b=-b;
    while (b){ i64 t=a%b; a=b; b=t; }
    return a;
}

struct Vec8 {
    std::array<i64, N> v{};
    bool operator==(const Vec8& o) const noexcept { return v == o.v; }
    bool operator<(const Vec8& o) const noexcept { return v < o.v; }
};
struct Vec8Hash {
    size_t operator()(const Vec8& a) const noexcept {
        u64 h = 1469598103934665603ull;
        for (i64 x : a.v){ h ^= (u64)x; h *= 1099511628211ull; }
        return (size_t)h;
    }
};

struct Mat {
    int n; i64 den; std::vector<i64> num;
    Mat() : n(0), den(1) {}
    Mat(int nn, i64 d) : n(nn), den(d), num(nn*nn, 0) {}
    i64& at(int i,int j){ return num[i*n+j]; }
    i64 at(int i,int j) const { return num[i*n+j]; }
};
struct MatHash {
    size_t operator()(const Mat& M) const noexcept {
        u64 h=1469598103934665603ull;
        h^=(u64)M.den; h*=1099511628211ull;
        for (i64 x:M.num){ h^=(u64)x; h*=1099511628211ull; }
        return (size_t)h;
    }
};
struct MatEq {
    bool operator()(const Mat& A,const Mat& B) const noexcept {
        return A.den==B.den && A.num==B.num;
    }
};

static void mat_reduce(Mat& M){
    i64 g = M.den<0 ? -M.den : M.den;
    for (i64 x : M.num){
        i64 ax = x<0 ? -x : x;
        if (ax){ g = gcd_i64(g, ax); if (g==1) break; }
    }
    if (g>1){ for (i64& x : M.num) x /= g; M.den /= g; }
    if (M.den<0){ M.den=-M.den; for (i64& x : M.num) x = -x; }
}
static Mat mat_identity(int n){
    Mat I(n, 1);
    for (int i=0; i<n; ++i) I.at(i,i)=1;
    return I;
}
static Mat mat_mul(const Mat& A, const Mat& B){
    int n=A.n; Mat C(n, A.den*B.den);
    for (int i=0;i<n;++i) for (int k=0;k<n;++k){
        i64 a=A.at(i,k); if (!a) continue;
        for (int j=0;j<n;++j) C.at(i,j) += a*B.at(k,j);
    }
    mat_reduce(C); return C;
}
static Mat reflection_matrix(const std::array<i64,N>& alpha){
    i64 nrm=0; for (int i=0;i<N;++i) nrm += alpha[i]*alpha[i];
    Mat M(N, nrm);
    for (int i=0;i<N;++i) for (int j=0;j<N;++j)
        M.at(i,j) = (i==j ? nrm : 0) - 2*alpha[i]*alpha[j];
    mat_reduce(M); return M;
}

struct VecQ {
    std::array<i64,N> num{}; i64 den=1;
    void reduce(){
        i64 g = den<0 ? -den : den;
        for (i64 x : num){
            i64 ax = x<0 ? -x : x;
            if (ax){ g=gcd_i64(g,ax); if (g==1) break; }
        }
        if (g>1){ for (i64& x : num) x /= g; den /= g; }
        if (den<0){ den=-den; for (i64& x : num) x=-x; }
    }
};
static VecQ apply_mat(const Mat& M, const VecQ& v){
    VecQ r; r.den=M.den*v.den;
    for (int i=0;i<N;++i){
        i64 s=0; for (int j=0;j<N;++j) s += M.at(i,j)*v.num[j];
        r.num[i]=s;
    }
    r.reduce(); return r;
}

static i64 det_int(std::vector<i64> A, int n){
    i64 sign=1, prev=1;
    for (int k=0;k<n;++k){
        if (A[k*n+k]==0){
            int r=-1;
            for (int i=k+1;i<n;++i) if (A[i*n+k]){ r=i; break; }
            if (r<0) return 0;
            for (int j=0;j<n;++j) std::swap(A[k*n+j], A[r*n+j]);
            sign = -sign;
        }
        for (int i=k+1;i<n;++i){
            for (int j=k+1;j<n;++j){
                i128 t = (i128)A[i*n+j]*A[k*n+k] - (i128)A[i*n+k]*A[k*n+j];
                t /= prev;
                A[i*n+j] = (i64)t;
            }
            A[i*n+k] = 0;
        }
        prev = A[k*n+k];
    }
    return sign * A[(n-1)*n+(n-1)];
}
static bool solve_rational(const std::vector<i64>& A_in, const std::vector<i64>& b_in,
                           int n, std::vector<i64>& num, i64& den){
    i64 D = det_int(A_in, n); if (D==0) return false;
    den = D; num.assign(n, 0);
    for (int col=0; col<n; ++col){
        std::vector<i64> Ai = A_in;
        for (int i=0;i<n;++i) Ai[i*n+col] = b_in[i];
        num[col] = det_int(Ai, n);
    }
    return true;
}
static int rank_int_general(std::vector<i64> A, int rows, int cols){
    int r=0; i64 prev=1; int row=0;
    for (int col=0; col<cols && row<rows; ++col){
        int piv=-1;
        for (int i=row;i<rows;++i) if (A[i*cols+col]){ piv=i; break; }
        if (piv<0) continue;
        if (piv!=row) for (int j=0;j<cols;++j) std::swap(A[row*cols+j], A[piv*cols+j]);
        for (int i=row+1;i<rows;++i){
            for (int j=col+1;j<cols;++j){
                i128 t = (i128)A[i*cols+j]*A[row*cols+col] - (i128)A[i*cols+col]*A[row*cols+j];
                t /= prev; A[i*cols+j] = (i64)t;
            }
            A[i*cols+col] = 0;
        }
        prev = A[row*cols+col]; ++row; ++r;
    }
    return r;
}

static std::array<std::array<i64,N>, L> simple_roots_E6_x2(){
    return {{
        {{ 1,-1,-1,-1,-1,-1,-1, 1}},
        {{ 2, 2, 0, 0, 0, 0, 0, 0}},
        {{-2, 2, 0, 0, 0, 0, 0, 0}},
        {{ 0,-2, 2, 0, 0, 0, 0, 0}},
        {{ 0, 0,-2, 2, 0, 0, 0, 0}},
        {{ 0, 0, 0,-2, 2, 0, 0, 0}},
    }};
}

struct FundamentalWeights { std::array<std::array<i64,N>, L> w; i64 den; };
static FundamentalWeights compute_fundamental_weights(
    const std::array<std::array<i64,N>, L>& s)
{
    std::vector<i64> G(L*L, 0);
    for (int i=0;i<L;++i) for (int j=0;j<L;++j){
        i64 ss=0; for (int k=0;k<N;++k) ss += s[i][k]*s[j][k];
        G[i*L+j] = ss;
    }
    for (i64& x : G) x /= 4;
    i64 detG = det_int(G, L);
    if (detG==0){ std::fprintf(stderr,"det(G)=0\n"); std::exit(1); }
    FundamentalWeights FW;
    FW.den = 2*detG;
    for (int i=0;i<L;++i) FW.w[i].fill(0);
    for (int i=0;i<L;++i){
        std::vector<i64> b(L, 0); b[i]=1;
        std::vector<i64> num; i64 den;
        solve_rational(G, b, L, num, den);
        i64 mult = detG / den;
        for (int k=0;k<L;++k){
            i64 coef = num[k]*mult;
            for (int c=0;c<N;++c) FW.w[i][c] += coef*s[k][c];
        }
    }
    return FW;
}

static std::vector<Mat> generate_weyl_group(
    const std::array<std::array<i64,N>, L>& s)
{
    std::vector<Mat> refs;
    for (int i=0;i<L;++i) refs.push_back(reflection_matrix(s[i]));
    Mat I = mat_identity(N);
    std::unordered_set<Mat, MatHash, MatEq> seen;
    seen.reserve(60000);
    seen.insert(I);
    std::vector<Mat> g; g.reserve(51840);
    g.push_back(I);
    size_t head=0;
    while (head < g.size()){
        Mat cur = g[head++];
        for (auto& r : refs){
            Mat ng = mat_mul(r, cur);
            if (!seen.count(ng)){ seen.insert(ng); g.push_back(std::move(ng)); }
        }
    }
    return g;
}

static Vec8 normalize_dir(const VecQ& q){
    i64 g=0;
    for (i64 x : q.num){
        i64 ax = x<0 ? -x : x;
        if (ax) g = g ? gcd_i64(g, ax) : ax;
    }
    Vec8 r;
    if (g==0){ r.v.fill(0); return r; }
    for (int i=0;i<N;++i) r.v[i] = q.num[i]/g;
    return r;
}
static Vec8 canonical_undirected(const Vec8& v){
    Vec8 r = v;
    for (int i=0;i<N;++i){
        if (r.v[i]>0) return r;
        if (r.v[i]<0){ for (int j=0;j<N;++j) r.v[j] = -r.v[j]; return r; }
    }
    return r;
}

static void kernel_5x6(const i64 M[K*L], std::array<i64, L>& c){
    for (int j=0;j<L;++j){
        i64 minor[K*K];
        for (int i=0;i<K;++i){
            int cc=0;
            for (int jj=0;jj<L;++jj){
                if (jj==j) continue;
                minor[i*K + cc++] = M[i*L + jj];
            }
        }
        std::vector<i64> mv(minor, minor+K*K);
        i64 d = det_int(mv, K);
        c[j] = ((j&1) ? -d : d);
    }
}

struct AlphaConeFilter {
    std::vector<i64> AtA;
    i64 detAtA_sign;
    std::vector<i64> A_simple;
    int n_dim;
    
    AlphaConeFilter(const std::array<std::array<i64,N>, L>& simple) : n_dim(N) {
        A_simple.assign(N*L, 0);
        for (int j=0; j<L; ++j)
            for (int i=0; i<N; ++i)
                A_simple[i*L + j] = simple[j][i];
        AtA.assign(L*L, 0);
        for (int i=0; i<L; ++i)
            for (int j=0; j<L; ++j){
                i64 s = 0;
                for (int k=0; k<N; ++k)
                    s += A_simple[k*L + i] * A_simple[k*L + j];
                AtA[i*L + j] = s;
            }
        i64 d = det_int(AtA, L);
        detAtA_sign = (d > 0) ? 1 : (d < 0 ? -1 : 0);
        if (detAtA_sign == 0){
            std::fprintf(stderr, "det(A^T A) = 0!\n"); std::exit(1);
        }
    }
    
    bool should_discard(const Vec8& h) const {
        std::array<i64, L> Atb;
        for (int j=0; j<L; ++j){
            i64 s = 0;
            for (int k=0; k<N; ++k)
                s += A_simple[k*L + j] * h.v[k];
            Atb[j] = s;
        }
        std::array<int, L> signs;
        for (int j=0; j<L; ++j){
            std::vector<i64> Mj = AtA;
            for (int i=0; i<L; ++i) Mj[i*L + j] = Atb[i];
            i64 d = det_int(Mj, L);
            signs[j] = (d > 0) ? 1 : (d < 0 ? -1 : 0);
            if (detAtA_sign < 0) signs[j] = -signs[j];
        }
        bool all_pos = true, all_neg = true;
        for (int j=0; j<L; ++j){
            if (signs[j] <= 0) all_pos = false;
            if (signs[j] >= 0) all_neg = false;
        }
        return all_pos || all_neg;
    }
};

static bool in_dom_closure(const Vec8& u,
                            const std::array<std::array<i64,N>, L>& simple_x2) {
    for (int i=0; i<L; ++i){
        i64 s = 0;
        for (int k=0; k<N; ++k) s += simple_x2[i][k] * u.v[k];
        if (s < 0) return false;
    }
    return true;
}

static void save_vectors_text(const std::string& path,
                               const std::vector<Vec8>& vecs)
{
    std::ofstream f(path);
    if (!f){ std::fprintf(stderr, "Не открылся %s\n", path.c_str()); std::exit(1); }
    f << vecs.size() << "\n";
    for (const auto& v : vecs){
        for (int k=0; k<N; ++k){
            if (k) f << " ";
            f << v.v[k];
        }
        f << "\n";
    }
}
static std::vector<Vec8> load_vectors_text(const std::string& path){
    std::ifstream f(path);
    if (!f){ std::fprintf(stderr, "Не открылся %s\n", path.c_str()); std::exit(1); }
    size_t cnt = 0; f >> cnt;
    std::vector<Vec8> vecs(cnt);
    for (size_t i=0; i<cnt; ++i){
        for (int k=0; k<N; ++k) f >> vecs[i].v[k];
    }
    return vecs;
}

// Бинарный формат для большого набора векторов (компактнее)
static void append_vectors_binary(std::ofstream& f,
                                   const Vec8* vecs, size_t n)
{
    f.write(reinterpret_cast<const char*>(vecs), n * sizeof(Vec8));
}


__device__ __forceinline__ unsigned mod_pow(unsigned a, unsigned e, unsigned p){
    unsigned long long r = 1, b = a;
    while (e){
        if (e&1) r = (r*b) % p;
        b = (b*b) % p;
        e >>= 1;
    }
    return (unsigned)r;
}
__device__ __forceinline__ unsigned mod_inv_dev(unsigned a, unsigned p){
    return mod_pow(a, p-2, p);
}
__device__ __forceinline__ bool rank_eq_5_dev(const long long M[K*L]){
    const unsigned p = (unsigned)PRIME;
    unsigned A[K*L];
    #pragma unroll
    for (int i=0; i<K*L; ++i){
        long long x = M[i] % (long long)p;
        if (x<0) x += p;
        A[i] = (unsigned)x;
    }
    int row = 0;
    #pragma unroll
    for (int col=0; col<L; ++col){
        if (row >= K) break;
        int piv = -1;
        for (int i=row; i<K; ++i){ if (A[i*L+col]){ piv=i; break; } }
        if (piv < 0) continue;
        if (piv != row){
            #pragma unroll
            for (int j=0; j<L; ++j){
                unsigned tmp = A[row*L+j]; A[row*L+j] = A[piv*L+j]; A[piv*L+j] = tmp;
            }
        }
        unsigned inv = mod_inv_dev(A[row*L+col], p);
        for (int i=row+1; i<K; ++i){
            unsigned f = A[i*L+col];
            if (!f) continue;
            unsigned long long factor = ((unsigned long long)f * inv) % p;
            for (int j=col; j<L; ++j){
                unsigned long long sub = (factor * A[row*L+j]) % p;
                unsigned v = (unsigned)((unsigned long long)A[i*L+j] + p - sub);
                if (v >= p) v -= p;
                A[i*L+j] = v;
            }
        }
        ++row;
    }
    return row == K;
}
__device__ __forceinline__ unsigned long long binC_d(
    const unsigned long long* C_table, int n, int k, int Nh)
{
    if (n < k || k < 0) return 0;
    if (k == 0) return 1;
    return C_table[n*6 + k];
}
__device__ void unrank_5_dev(unsigned long long idx, int Nh,
                             const unsigned long long* C_table,
                             int& i0, int& i1, int& i2, int& i3, int& i4)
{
    int idxs[5] = {0,0,0,0,0};
    int prev = -1;
    for (int pos = 0; pos < 5; ++pos){
        int k_remain = 4 - pos;
        int candidate = prev + 1;
        while (candidate < Nh){
            unsigned long long cnt = binC_d(C_table, Nh - candidate - 1, k_remain, Nh);
            if (idx < cnt){
                idxs[pos] = candidate;
                prev = candidate;
                break;
            } else {
                idx -= cnt;
                ++candidate;
            }
        }
    }
    i0 = idxs[0]; i1 = idxs[1]; i2 = idxs[2]; i3 = idxs[3]; i4 = idxs[4];
}
__global__ void filter_kernel(
    const long long* __restrict__ HB,
    const unsigned long long* __restrict__ C_table,
    int Nh,
    unsigned long long total_combos,
    unsigned long long base_index,
    unsigned long long batch_size,
    unsigned long long* __restrict__ out_buf,
    unsigned long long* __restrict__ out_count,
    unsigned long long out_capacity,
    int* __restrict__ overflow_flag)
{
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;
    unsigned long long idx = base_index + tid;
    if (idx >= total_combos) return;

    int i0,i1,i2,i3,i4;
    unrank_5_dev(idx, Nh, C_table, i0, i1, i2, i3, i4);

    long long M[K*L];
    #pragma unroll
    for (int j=0;j<L;++j){
        M[0*L+j] = HB[i0*L+j];
        M[1*L+j] = HB[i1*L+j];
        M[2*L+j] = HB[i2*L+j];
        M[3*L+j] = HB[i3*L+j];
        M[4*L+j] = HB[i4*L+j];
    }

    if (!rank_eq_5_dev(M)) return;

    unsigned long long code =
        ((unsigned long long)i0)
      | ((unsigned long long)i1 << 12)
      | ((unsigned long long)i2 << 24)
      | ((unsigned long long)i3 << 36)
      | ((unsigned long long)i4 << 48);

    unsigned long long pos = atomicAdd((unsigned long long*)out_count, 1ull);
    if (pos < out_capacity){
        out_buf[pos] = code;
    } else {
        atomicExch(overflow_flag, 1);
    }
}

static void weyl_extend_streaming(
    const std::vector<Vec8>& partial,
    const std::vector<Mat>& W,
    const std::string& output_path,
    size_t chunk_size = 1'000'000)
{
    std::printf("    Поточное разнесение в файл %s\n", output_path.c_str());
    std::printf("    |partial| = %zu, |W| = %zu, ожидается до %zu вставок\n",
                partial.size(), W.size(), partial.size() * W.size());
    
    std::ofstream fout(output_path, std::ios::binary);
    if (!fout){
        std::fprintf(stderr, "Не открылся %s\n", output_path.c_str());
        std::exit(1);
    }
    
    std::vector<Vec8> buffer;
    buffer.reserve(chunk_size);
    
    auto t0 = std::chrono::steady_clock::now();
    size_t total_written = 0;
    
    for (size_t pi = 0; pi < partial.size(); ++pi){
        const auto& u = partial[pi];
        VecQ uq;
        uq.den = 1;
        for (int k=0; k<N; ++k) uq.num[k] = u.v[k];
        
        for (const auto& M : W){
            VecQ wv = apply_mat(M, uq);
            Vec8 nv = normalize_dir(wv);
            buffer.push_back(nv);
            
            if (buffer.size() >= chunk_size){
                append_vectors_binary(fout, buffer.data(), buffer.size());
                total_written += buffer.size();
                buffer.clear();
            }
        }
        
        // Прогресс каждые 1000 partial-векторов
        if ((pi + 1) % 1000 == 0 || pi + 1 == partial.size()){
            auto t1 = std::chrono::steady_clock::now();
            double el = std::chrono::duration<double>(t1-t0).count();
            double frac = (double)(pi+1)/(double)partial.size();
            double eta = frac>0 ? el*(1.0/frac - 1.0) : 0;
            std::fprintf(stderr,
                "    [%.0f с] разнесено %zu/%zu (%.2f%%), записано %zu, ETA %.0f с\n",
                el, pi+1, partial.size(), 100.0*frac,
                total_written + buffer.size(), eta);
        }
    }
    
    if (!buffer.empty()){
        append_vectors_binary(fout, buffer.data(), buffer.size());
        total_written += buffer.size();
        buffer.clear();
    }
    fout.close();
    
    auto t1 = std::chrono::steady_clock::now();
    std::printf("    Записано %zu векторов в %s за %.1f с\n",
                total_written, output_path.c_str(),
                std::chrono::duration<double>(t1-t0).count());
}

static size_t external_sort_dedup(const std::string& input_path,
                                   const std::string& output_path,
                                   size_t chunk_vectors = 8'000'000)
{
    std::printf("    Внешняя дедупликация %s -> %s (chunks по %zu)\n",
                input_path.c_str(), output_path.c_str(), chunk_vectors);
    
    std::ifstream fin(input_path, std::ios::binary);
    if (!fin){
        std::fprintf(stderr, "Не открылся %s\n", input_path.c_str());
        std::exit(1);
    }
    
    std::vector<std::string> run_files;
    std::vector<Vec8> buf;
    buf.resize(chunk_vectors);
    
    int run_idx = 0;
    while (true){
        fin.read(reinterpret_cast<char*>(buf.data()), chunk_vectors * sizeof(Vec8));
        std::streamsize bytes_read = fin.gcount();
        if (bytes_read == 0) break;
        size_t n_read = bytes_read / sizeof(Vec8);
        
        // sort + uniq
        std::sort(buf.begin(), buf.begin() + n_read);
        size_t n_uniq = std::unique(buf.begin(), buf.begin() + n_read) - buf.begin();
        
        std::string run_path = input_path + ".run" + std::to_string(run_idx++);
        std::ofstream frun(run_path, std::ios::binary);
        frun.write(reinterpret_cast<const char*>(buf.data()), n_uniq * sizeof(Vec8));
        frun.close();
        run_files.push_back(run_path);
        
        std::fprintf(stderr, "    [run %d] прочитано %zu, осталось %zu уникальных\n",
                     run_idx, n_read, n_uniq);
        
        if (n_read < chunk_vectors) break;
    }
    fin.close();
    
    std::printf("    Создано %zu run-файлов, выполняем k-way merge с дедупом...\n",
                run_files.size());
    
 
    struct RunReader {
        std::ifstream f;
        Vec8 cur;
        bool has_next;
        std::string path;
        
        RunReader(const std::string& p) : path(p) {
            f.open(p, std::ios::binary);
            advance();
        }
        void advance(){
            f.read(reinterpret_cast<char*>(&cur), sizeof(Vec8));
            has_next = (f.gcount() == sizeof(Vec8));
        }
    };
    std::vector<std::unique_ptr<RunReader>> readers;
    for (const auto& p : run_files) readers.emplace_back(new RunReader(p));
    
    std::ofstream fout(output_path, std::ios::binary);
    if (!fout){
        std::fprintf(stderr, "Не открылся %s\n", output_path.c_str());
        std::exit(1);
    }
    
    Vec8 last; bool has_last = false;
    size_t total_uniq = 0;
    
    while (true){
        int best = -1;
        for (size_t i=0; i<readers.size(); ++i){
            if (!readers[i]->has_next) continue;
            if (best < 0 || readers[i]->cur < readers[best]->cur) best = (int)i;
        }
        if (best < 0) break;
        
        Vec8 v = readers[best]->cur;
        if (!has_last || !(v == last)){
            fout.write(reinterpret_cast<const char*>(&v), sizeof(Vec8));
            ++total_uniq;
            last = v;
            has_last = true;
        }
        readers[best]->advance();
    }
    
    fout.close();
    readers.clear();
    
    for (const auto& p : run_files) std::remove(p.c_str());
    
    std::printf("    Готово, уникальных: %zu\n", total_uniq);
    return total_uniq;
}


int main(int argc, char** argv){
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    auto t_start = std::chrono::steady_clock::now();
    unsigned long long max_out = 32000000ull;
    bool dom_filter = true;
    bool no_extend = false;
    std::string resume_path = "";
    
    for (int i=1; i<argc; ++i){
        std::string a = argv[i];
        if (a=="--maxout" && i+1<argc){
            max_out = std::strtoull(argv[i+1], nullptr, 10); ++i;
        } else if (a=="--no-dom-filter"){
            dom_filter = false;
        } else if (a=="--no-extend"){
            no_extend = true;
        } else if (a=="--resume" && i+1<argc){
            resume_path = argv[i+1]; ++i;
        }
    }

    auto simple = simple_roots_E6_x2();
    
    std::printf("[1] Группа Вейля...\n");
    auto W = generate_weyl_group(simple);
    std::printf("    |W| = %zu\n", W.size());
    if (W.size() != 51840){ std::fprintf(stderr,"|W|≠51840\n"); return 1; }

    std::vector<Vec8> partial_vecs;

    if (!resume_path.empty()){
        std::printf("\n[*] RESUME режим: загружаем partial из %s\n",
                    resume_path.c_str());
        partial_vecs = load_vectors_text(resume_path);
        std::printf("    Загружено %zu векторов\n", partial_vecs.size());
    } else {
        int dev_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&dev_count));
        if (dev_count == 0){ std::fprintf(stderr, "Нет CUDA\n"); return 1; }
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::printf("GPU 0: %s (CC %d.%d, %d SMs, %.2f GB)\n",
                    prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                    prop.totalGlobalMem / (1024.0*1024.0*1024.0));
        std::printf("max_out: %llu, dom_filter: %s, no_extend: %s\n",
                    max_out, dom_filter?"ВКЛ":"ВЫКЛ", no_extend?"ДА":"НЕТ");

        std::printf("\n[2] Фунд. веса...\n");
        auto FW = compute_fundamental_weights(simple);

        std::printf("[3] Сбор гиперплоскостей...\n");
        std::array<VecQ, L> dom;
        for (int i=0;i<L;++i){
            for (int j=0;j<N;++j) dom[i].num[j] = FW.w[i][j];
            dom[i].den = FW.den;
            dom[i].reduce();
        }
        std::unordered_set<Vec8, Vec8Hash> hyp_set;
        hyp_set.reserve(2048);
        for (size_t g=0; g<W.size(); ++g)
            for (int i=0;i<L;++i){
                VecQ v = apply_mat(W[g], dom[i]);
                hyp_set.insert(canonical_undirected(normalize_dir(v)));
            }
        std::vector<Vec8> H_full(hyp_set.begin(), hyp_set.end());
        std::sort(H_full.begin(), H_full.end(),
                  [](const Vec8& a, const Vec8& b){ return a.v < b.v; });
        std::printf("    |H| (до фильтра) = %zu\n", H_full.size());

        std::printf("\n[3.5] Фильтр идеи 1...\n");
        AlphaConeFilter af(simple);
        std::vector<Vec8> H;
        H.reserve(H_full.size());
        size_t n_discarded = 0;
        for (const auto& h : H_full){
            if (af.should_discard(h)) ++n_discarded;
            else H.push_back(h);
        }
        std::printf("    выкинуто: %zu, осталось: %zu\n", n_discarded, H.size());
        const size_t Nh = H.size();
        if (Nh < 5){ std::fprintf(stderr, "Мало векторов\n"); return 1; }

        std::printf("\n[4] Базис и проекции...\n");
        std::vector<std::array<i64,N>> basis_rows;
        for (const auto& h : H){
            std::vector<i64> test;
            for (auto& r : basis_rows) for (i64 x : r) test.push_back(x);
            for (i64 x : h.v) test.push_back(x);
            int rk = rank_int_general(test, (int)basis_rows.size()+1, N);
            if (rk > (int)basis_rows.size()){
                std::array<i64,N> a; std::copy(h.v.begin(), h.v.end(), a.begin());
                basis_rows.push_back(a);
                if ((int)basis_rows.size()==L) break;
            }
        }
        if ((int)basis_rows.size()!=L){
            std::fprintf(stderr,"Добираем базис из H_full...\n");
            basis_rows.clear();
            for (const auto& h : H_full){
                std::vector<i64> test;
                for (auto& r : basis_rows) for (i64 x : r) test.push_back(x);
                for (i64 x : h.v) test.push_back(x);
                int rk = rank_int_general(test, (int)basis_rows.size()+1, N);
                if (rk > (int)basis_rows.size()){
                    std::array<i64,N> a; std::copy(h.v.begin(), h.v.end(), a.begin());
                    basis_rows.push_back(a);
                    if ((int)basis_rows.size()==L) break;
                }
            }
            if ((int)basis_rows.size()!=L){
                std::fprintf(stderr,"FAIL\n"); return 1;
            }
        }

        std::vector<i64> HB_host(Nh*L);
        for (size_t i=0; i<Nh; ++i)
            for (int j=0; j<L; ++j){
                i64 s = 0;
                for (int k=0; k<N; ++k) s += H[i].v[k] * basis_rows[j][k];
                HB_host[i*L+j] = s;
            }

        std::vector<unsigned long long> C_table((Nh+1)*6, 0);
        for (int n=0; n<=(int)Nh; ++n){
            C_table[n*6+0] = 1;
            for (int k=1; k<=5; ++k){
                if (k>n) C_table[n*6+k] = 0;
                else C_table[n*6+k] = C_table[(n-1)*6+(k-1)] + C_table[(n-1)*6+k];
            }
        }
        unsigned long long total_combos = C_table[Nh*6 + 5];
        std::printf("    C(%zu, 5) = %llu\n", Nh, total_combos);

        long long* d_HB = nullptr;
        unsigned long long* d_C = nullptr;
        unsigned long long* d_out = nullptr;
        unsigned long long* d_count = nullptr;
        int* d_overflow = nullptr;

        CUDA_CHECK(cudaMalloc((void**)&d_HB,  Nh*L*sizeof(long long)));
        CUDA_CHECK(cudaMalloc((void**)&d_C,   (Nh+1)*6*sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc((void**)&d_out, max_out*sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc((void**)&d_count, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc((void**)&d_overflow, sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_HB, HB_host.data(),
            Nh*L*sizeof(long long), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C, C_table.data(),
            (Nh+1)*6*sizeof(unsigned long long), cudaMemcpyHostToDevice));

        int n_workers = (int)std::thread::hardware_concurrency();
        if (n_workers <= 0) n_workers = 4;
        if (const char* s = std::getenv("E6_CPU_WORKERS")){
            int t = std::atoi(s); if (t > 0) n_workers = t;
        }
        std::printf("\n[5] GPU/CPU pipeline (workers=%d)\n", n_workers);
        
        std::vector<std::unordered_set<Vec8, Vec8Hash>> worker_dom(n_workers);
        std::vector<std::unordered_set<Vec8, Vec8Hash>> worker_full(n_workers);
        for (auto& s : worker_dom) s.reserve(1024);
        for (auto& s : worker_full) s.reserve(1024);

        std::ofstream partial_dump_dom("partial_dump_dom.bin", std::ios::binary);
        std::ofstream partial_dump_full("partial_dump_full.bin", std::ios::binary);
        size_t total_dumped_dom = 0, total_dumped_full = 0;
        
        const size_t flush_threshold_per_worker = 50'000;

        auto flush_workers = [&](){
            // Сливаем worker_dom -> partial_dump_dom
            for (auto& s : worker_dom){
                if (s.empty()) continue;
                std::vector<Vec8> buf(s.begin(), s.end());
                partial_dump_dom.write(
                    reinterpret_cast<const char*>(buf.data()),
                    buf.size() * sizeof(Vec8));
                total_dumped_dom += buf.size();
                s.clear();
            }
            for (auto& s : worker_full){
                if (s.empty()) continue;
                std::vector<Vec8> buf(s.begin(), s.end());
                partial_dump_full.write(
                    reinterpret_cast<const char*>(buf.data()),
                    buf.size() * sizeof(Vec8));
                total_dumped_full += buf.size();
                s.clear();
            }
        };

        auto process_codes = [&](const unsigned long long* codes, size_t n,
                                 int worker_id)
        {
            auto& cs_dom  = worker_dom[worker_id];
            auto& cs_full = worker_full[worker_id];
            for (size_t idx = 0; idx < n; ++idx){
                unsigned long long code = codes[idx];
                int i0 = (int)( code        & 0xFFFull);
                int i1 = (int)((code >> 12) & 0xFFFull);
                int i2 = (int)((code >> 24) & 0xFFFull);
                int i3 = (int)((code >> 36) & 0xFFFull);
                int i4 = (int)((code >> 48) & 0xFFFull);

                i64 M[K*L];
                for (int j=0; j<L; ++j){
                    M[0*L+j] = HB_host[i0*L+j];
                    M[1*L+j] = HB_host[i1*L+j];
                    M[2*L+j] = HB_host[i2*L+j];
                    M[3*L+j] = HB_host[i3*L+j];
                    M[4*L+j] = HB_host[i4*L+j];
                }
                std::vector<i64> Mv(M, M+K*L);
                if (rank_int_general(Mv, K, L) < K) continue;

                std::array<i64, L> c;
                kernel_5x6(M, c);

                std::array<i64, N> v_full{};
                for (int j=0; j<L; ++j){
                    if (!c[j]) continue;
                    for (int k=0; k<N; ++k) v_full[k] += basis_rows[j][k]*c[j];
                }
                bool zero=true;
                for (int k=0;k<N;++k) if (v_full[k]){ zero=false; break; }
                if (zero) continue;

                i64 g = 0;
                for (i64 x : v_full){
                    i64 ax = x<0 ? -x : x;
                    if (ax) g = g ? gcd_i64(g, ax) : ax;
                }
                Vec8 pos, neg;
                for (int k=0; k<N; ++k){
                    pos.v[k] = v_full[k]/g;
                    neg.v[k] = -pos.v[k];
                }
                
                if (dom_filter){
                    if (in_dom_closure(pos, simple)) cs_dom.insert(pos);
                    if (in_dom_closure(neg, simple)) cs_dom.insert(neg);
                } else {
                    cs_full.insert(pos);
                    cs_full.insert(neg);
                }
            }
        };

        unsigned long long* h_pinned = nullptr;
        cudaError_t pin_err = cudaMallocHost((void**)&h_pinned,
            max_out * sizeof(unsigned long long));
        if (pin_err != cudaSuccess){
            std::fprintf(stderr, "cudaMallocHost: %s\n",
                         cudaGetErrorString(pin_err));
            h_pinned = (unsigned long long*)std::malloc(
                max_out * sizeof(unsigned long long));
            if (!h_pinned){ std::fprintf(stderr,"OOM\n"); return 3; }
        }

        unsigned long long batch_combos = 1ull << 26;
        if (const char* s = std::getenv("E6_BATCH")){
            unsigned long long b = std::strtoull(s, nullptr, 10);
            if (b > 0) batch_combos = b;
        }
        const int block = 256;
        unsigned long long min_batch = 1ull << 20;

        unsigned long long processed = 0;
        unsigned long long total_passed = 0;
        int batch_id = 0;

        auto t_loop = std::chrono::steady_clock::now();

        while (processed < total_combos){
            unsigned long long this_batch = std::min(batch_combos,
                                                       total_combos - processed);
            CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned long long)));
            CUDA_CHECK(cudaMemset(d_overflow, 0, sizeof(int)));

            unsigned long long grid = (this_batch + block - 1) / block;
            filter_kernel<<<(unsigned)grid, block>>>(
                d_HB, d_C, (int)Nh, total_combos,
                processed, this_batch,
                d_out, d_count, max_out, d_overflow);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            unsigned long long cur_count = 0;
            int overflow = 0;
            CUDA_CHECK(cudaMemcpy(&cur_count, d_count,
                sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&overflow, d_overflow,
                sizeof(int), cudaMemcpyDeviceToHost));

            if (overflow){
                if (batch_combos <= min_batch){
                    std::fprintf(stderr,"OVERFLOW even at min batch\n");
                    return 2;
                }
                batch_combos = std::max(batch_combos/2, min_batch);
                std::fprintf(stderr, "  overflow, batch -> %llu\n", batch_combos);
                continue;
            }

            if (cur_count > 0){
                CUDA_CHECK(cudaMemcpy(h_pinned, d_out,
                    cur_count * sizeof(unsigned long long),
                    cudaMemcpyDeviceToHost));
                std::vector<std::thread> workers;
                workers.reserve(n_workers);
                unsigned long long chunk = (cur_count + n_workers - 1) / n_workers;
                for (int w = 0; w < n_workers; ++w){
                    unsigned long long start = w * chunk;
                    unsigned long long end = std::min(start + chunk, cur_count);
                    if (start >= end) break;
                    workers.emplace_back([&, start, end, w](){
                        process_codes(h_pinned + start, end - start, w);
                    });
                }
                for (auto& th : workers) th.join();
            }
            
            size_t max_worker = 0;
            for (auto& s : worker_dom) max_worker = std::max(max_worker, s.size());
            for (auto& s : worker_full) max_worker = std::max(max_worker, s.size());
            if (max_worker >= flush_threshold_per_worker){
                flush_workers();
            }

            processed += this_batch;
            total_passed += cur_count;
            ++batch_id;
            
            auto tn = std::chrono::steady_clock::now();
            double el = std::chrono::duration<double>(tn-t_loop).count();
            double frac = (double)processed/(double)total_combos;
            double eta = frac>0 ? el*(1.0/frac - 1.0) : 0;
            std::fprintf(stderr,
                "  [b%d] %.3f%% (%.0fс ETA %.0fс) cur_pass=%llu total_pass=%llu "
                "dumped_dom=%zu dumped_full=%zu\n",
                batch_id, 100.0*frac, el, eta,
                cur_count, total_passed, total_dumped_dom, total_dumped_full);
        }

        flush_workers();
        partial_dump_dom.close();
        partial_dump_full.close();

        auto t_loop_end = std::chrono::steady_clock::now();
        std::printf("\nGPU/CPU pipeline завершён за %.1f с\n",
            std::chrono::duration<double>(t_loop_end-t_loop).count());
        std::printf("Прошло rank-фильтр: %llu\n", total_passed);
        std::printf("Сброшено в partial_dump_dom.bin: %zu, "
                    "в partial_dump_full.bin: %zu\n",
                    total_dumped_dom, total_dumped_full);

        cudaFree(d_HB); cudaFree(d_C); cudaFree(d_out);
        cudaFree(d_count); cudaFree(d_overflow);
        if (pin_err == cudaSuccess) cudaFreeHost(h_pinned);
        else std::free(h_pinned);

        std::printf("\n[6] Дедупликация partial...\n");
        std::string src = dom_filter ? "partial_dump_dom.bin"
                                     : "partial_dump_full.bin";
        std::string dst = "partial_dedup.bin";
        size_t n_uniq = external_sort_dedup(src, dst);
        std::printf("    Уникальных partial-векторов: %zu\n", n_uniq);
        
        partial_vecs.reserve(n_uniq);
        std::ifstream fdedup(dst, std::ios::binary);
        Vec8 v;
        while (fdedup.read(reinterpret_cast<char*>(&v), sizeof(Vec8))){
            partial_vecs.push_back(v);
        }
        fdedup.close();
        std::remove(dst.c_str());
        std::remove(src.c_str());

        save_vectors_text("partial_in_C.txt", partial_vecs);
        std::printf("    Сохранено в partial_in_C.txt - можно использовать --resume\n");

        std::printf("\n|U \\cap \\bar C| (или весь partial если --no-dom-filter) = %zu\n",
                    partial_vecs.size());
    }

    if (no_extend){
        std::printf("\n[--no-extend] Завершаем без разнесения W.\n");
        std::printf("Файл partial_in_C.txt содержит %zu векторов.\n",
                    partial_vecs.size());
        auto t_end = std::chrono::steady_clock::now();
        std::printf("Общее время: %.1f с\n",
            std::chrono::duration<double>(t_end-t_start).count());
        return 0;
    }

    std::printf("\n[7] Потоковое разнесение группой Вейля...\n");
    weyl_extend_streaming(partial_vecs, W, "weyl_extended_raw.bin");

    std::printf("\n[8] Дедупликация полного характеристического набора...\n");
    size_t n_charset = external_sort_dedup("weyl_extended_raw.bin",
                                            "characteristic_set.bin");
    std::printf("    |U| = %zu\n", n_charset);
   
    std::remove("weyl_extended_raw.bin");

    std::printf("\n[9] Запись текстовых файлов...\n");
    std::ifstream fin("characteristic_set.bin", std::ios::binary);
    std::ofstream f1("e6_characteristic_set_rational.txt");
    std::ofstream f2("e6_characteristic_set_fractions.txt");
    f1.setf(std::ios::fixed); f1.precision(10);
    Vec8 v;
    while (fin.read(reinterpret_cast<char*>(&v), sizeof(Vec8))){
        for (int k=0; k<N; ++k){ if (k) f1 << "  "; f1 << (double)v.v[k]; }
        f1 << "\n";
        for (int k=0; k<N; ++k){ if (k) f2 << "  "; f2 << v.v[k]; }
        f2 << "\n";
    }
    f1.close(); f2.close(); fin.close();

    auto t_end = std::chrono::steady_clock::now();
    std::printf("\nГотово. |U| = %zu\n", n_charset);
    std::printf("Общее время: %.1f с\n",
        std::chrono::duration<double>(t_end-t_start).count());
    return 0;
}
