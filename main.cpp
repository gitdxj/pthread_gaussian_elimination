#include <iostream>
#include "matrix.h"
#include "pthread_lu.h"
using namespace std;

void *hello(void *arg);



int main()
{
    long long a;
    test_lu(1000, 8, a);
    cout << a;
    return 0;
}


/**测试多线程的hello函数**/
void *hello(void *arg)
{
    int *thread_No = (int*) arg;
//    cout<<"this is thread"<<*thread_No<<endl;
    printf("Hello from thread %d \n", *thread_No);
}






