#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& a, const std::vector<float>& b, int n) {
	std::vector<float> result(n * n);
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < n; ++i)
		{
			auto* currentRow = &result[i * n];
			for (int j = 0; j < n; ++j)
			{
				auto* currentColum = &b[j * n];
				float element = a[i * n + j];
				for (int k = 0; k < n; ++k)
				{
					currentRow[k] += element * currentColum[k];
				}
			}
		}
	}
	return result;
}