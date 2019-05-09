#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED
/**此文件中是一些矩阵操作的基本方法，如矩阵初始化，向控制台打印等，函数定义在对应的cpp文件中**/
#include <iostream>
#include <cstdlib>  // 用来产生随机数
#include <iomanip>  // 用来格式化输出
#include <algorithm>  // 需要里面的swap函数
#include <time.h>  // 需要时间作种子
#include <x86intrin.h>

void matrix_initialize(float **Matrix, int N, int scale = 10);  // 为矩阵元素赋随机值，范围默认取0~10
void copy_matrix(float** dst, float** src, int N);  // 将src矩阵的值copy到dst中
void show_matrix(float **Matrix, int N);  // 在控制台打印矩阵
bool swap_rows(float **Matrix, int N, int k);  // 当使用高斯消去法时若A(k,k)为0则从下面的行中找出从k列不为0的一行和k行互换
void gaussian_elimination_lu(float**Matrix, int N);  // 标准的LU高斯消去法算法
void gaussian_elimination_sse_unaligned(float **Matrix, int N, bool p45=true);
void gaussian_elimination_avx_unaligned(float **Matrix, int N, bool p45=true);
#endif // MATRIX_H_INCLUDED
