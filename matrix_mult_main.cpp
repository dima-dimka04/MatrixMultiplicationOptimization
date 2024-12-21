#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <string>

#define MKL
#ifdef MKL
#include "mkl.h"
#endif

using namespace std;


#include <immintrin.h>

//icx /O2 /Qmkl /QxHost /Qopenmp matrix_mult_main.cpp -o matrix_mult_main.exe
// works good but 2 load 1 fmadd
void matrix_mult_optimized2(double* a, double* b, double* res, size_t size) {
    constexpr int BLOCK_SIZE = 32;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < size; i += BLOCK_SIZE) {
        for (int j = 0; j < size; j += BLOCK_SIZE) {
            for (int k = 0; k < size; k += BLOCK_SIZE) {
                for (int ii = i; ii < i + BLOCK_SIZE && ii < size; ++ii) {
                    for (int kk = k; kk < k + BLOCK_SIZE && kk < size; ++kk) {
                        double tmp = a[ii * size + kk];
                        int min_jj = std::min(j + BLOCK_SIZE, static_cast<int>(size));

                        int jj = j;
                        __m256d vec_tmp = _mm256_set1_pd(tmp);
                        for (; jj + 4 <= min_jj; jj += 4) {
                            __m256d vec_b = _mm256_loadu_pd(&b[kk * size + jj]);
                            __m256d vec_res = _mm256_loadu_pd(&res[ii * size + jj]);

                            vec_res = _mm256_fmadd_pd(vec_tmp, vec_b, vec_res);

                            _mm256_storeu_pd(&res[ii * size + jj], vec_res);
                        }
                        for (; jj < min_jj; ++jj) {
                            res[ii * size + jj] += tmp * b[kk * size + jj];
                        }
                    }
                }
            }
        }
    }
}

// works harder
void matrix_mult_optimized1(double* a, double* b, double* res, size_t size) { 
    constexpr size_t BLOCK_SIZE = 32;

    #pragma omp parallel for schedule(dynamic)
    for (size_t bi = 0; bi < size; bi += BLOCK_SIZE) {
        for (size_t bj = 0; bj < size; bj += BLOCK_SIZE) {
            for (size_t bk = 0; bk < size; bk += BLOCK_SIZE) {
                for (size_t i = bi; i < std::min(bi + BLOCK_SIZE, size); ++i) {
                    for (size_t j = bj; j < std::min(bj + BLOCK_SIZE, size); ++j) {
                        double sum = 0.0;
                        size_t k = bk;

                        __m256d sum_vec = _mm256_setzero_pd();
                        for (; k + 4 <= std::min(bk + BLOCK_SIZE, size); k += 4) {
                            __m256d vec_a = _mm256_loadu_pd(&a[i * size + k]);
                            __m256d vec_b = _mm256_set_pd(
                                b[(k + 3) * size + j],
                                b[(k + 2) * size + j],
                                b[(k + 1) * size + j],
                                b[k * size + j]
                            );
                            sum_vec = _mm256_fmadd_pd(vec_a, vec_b, sum_vec);
                        }
                        double temp[4];
                        _mm256_storeu_pd(temp, sum_vec);
                        sum += temp[0] + temp[1] + temp[2] + temp[3];
                        for (; k < std::min(bk + BLOCK_SIZE, size); ++k) {
                            sum += a[i * size + k] * b[k * size + j];
                        }

                        res[i * size + j] += sum;
                    }
                }
            }
        }
    }
}



void generation(double * mat, size_t size)
{
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> uniform_distance(-2.001, 2.001);
	for (size_t i = 0; i < size * size; i++)
		mat[i] = uniform_distance(gen);
}

void matrix_mult(double * a, double * b, double * res, size_t size)
{
#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
	    {
			for (int k = 0; k < size; k++)
		    {
				res[i*size + j] += a[i*size + k] * b[k*size + j];
			}
		}
	}
}

int main()
{
	double *mat, *mat_mkl, *a, *b, *a_mkl, *b_mkl;
	size_t size = 1001;
	chrono::time_point<chrono::system_clock> start, end;

	mat = new double[size * size];
	a = new double[size * size];
	b = new double[size * size];
	generation(a, size);
	generation(b, size);
	memset(mat, 0, size*size * sizeof(double));

#ifdef MKL     
    mat_mkl = new double[size * size];
	a_mkl = new double[size * size];
	b_mkl = new double[size * size];
	memcpy(a_mkl, a, sizeof(double)*size*size);
    	memcpy(b_mkl, b, sizeof(double)*size*size);
	memset(mat_mkl, 0, size*size * sizeof(double));
#endif

	start = chrono::system_clock::now();
	matrix_mult_optimized2(a, b, mat, size);
	end = chrono::system_clock::now();
    
   
	int elapsed_seconds = chrono::duration_cast<chrono::milliseconds>
		(end - start).count();
	cout << "Total time: " << elapsed_seconds/1000.0 << " sec" << endl;

#ifdef MKL 
	start = chrono::system_clock::now();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, size, size, 1.0, a_mkl, size, b_mkl, size, 0.0, mat_mkl, size);
    end = chrono::system_clock::now();
    
    elapsed_seconds = chrono::duration_cast<chrono::milliseconds>
		(end - start).count();
	cout << "Total time mkl: " << elapsed_seconds/1000.0 << " sec" << endl;
     
    int flag = 0;
    for (unsigned int i = 0; i < size * size; i++)
        if(abs(mat[i] - mat_mkl[i]) > size*1e-14){
		    flag = 1;
        }
    if (flag)
        cout << "fail" << endl;
    else
        cout << "correct" << endl; 
    
    delete (a_mkl);
    delete (b_mkl);
    delete (mat_mkl);
#endif

    delete (a);
    delete (b);
    delete (mat);

	//system("pause");
	
	return 0;
}
