#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED
/**���ļ�����һЩ��������Ļ���������������ʼ���������̨��ӡ�ȣ����������ڶ�Ӧ��cpp�ļ���**/
#include <iostream>
#include <cstdlib>  // �������������
#include <iomanip>  // ������ʽ�����
#include <algorithm>  // ��Ҫ�����swap����
#include <time.h>  // ��Ҫʱ��������

void matrix_initialize(float **Matrix, int N, int scale = 10);  // Ϊ����Ԫ�ظ����ֵ����ΧĬ��ȡ0~10
void copy_matrix(float** dst, float** src, int N);  // ��src�����ֵcopy��dst��
void show_matrix(float **Matrix, int N);  // �ڿ���̨��ӡ����
bool swap_rows(float **Matrix, int N, int k);  // ��ʹ�ø�˹��ȥ��ʱ��A(k,k)Ϊ0�������������ҳ���k�в�Ϊ0��һ�к�k�л���
void LU(float**Matrix, int N);  // ��׼��LU��˹��ȥ���㷨

#endif // MATRIX_H_INCLUDED
