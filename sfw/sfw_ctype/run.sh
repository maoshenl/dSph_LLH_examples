#g++ functions.cpp -lgsl -lgslcblas
#g++ test.cpp
#time ./a.out

g++  -shared -o sfw_function.so -fPIC sfw_function.cpp -lgsl -lgslcblas
time python test_ctype.py 

#python test.py
