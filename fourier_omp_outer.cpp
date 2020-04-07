/*
 * Copyright (C) 2020 zampaze@miamioh.edu
 */

/* 
 * File:   fourier_omp_outer.cpp
 * Author: zampaze
 *
 * Created on April 7, 2020, 2:14 PM
 */



#include <cstdlib>
#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

    
/**
 * Initialize input arrays to emulate complex numbers
 * @param N
 * @param outReal
 * @param outImagine
 */
void initialize(int N, double oR[], double oI[]) {
  for (int n = 0; n < N; n++) {
    double t = 2.0*M_PI*(n - N/2);
    oR[n] = sin(100.00*t/N);
    oI[n] = 0.0;
  }
}

/**
 * Compute the Discrete Fourier Transform (DFT) of a given set of real and 
 * imaginary number arrays.
 * Assumes every array is the same length
 * @param inRealNums
 * @param inImagineNums
 * @param outRealNums
 * @param inImagineNums
 */
void calculateDFT(int N, double inRealNums[], double inImagineNums[],
        double outRealNums[], double outImagineNums[]) {
    // assign a value to every Output element
#pragma omp parallel for 
    for (int outerIter = 0; outerIter < N; outerIter++) {
        double sumReal = 0;
        double sumImagine = 0;

        // process through each Input element
        for (int innerIter = 0; innerIter < N; innerIter++) {
            // begin the summation for the current term

            double angle = (2.0 * M_PI) / N * (innerIter * outerIter);
            sumReal += inRealNums[innerIter] * cos(angle)
                    + (-1 * inImagineNums[innerIter] * sin(angle));
            sumImagine += -inRealNums[innerIter] * sin(angle)
                    + inImagineNums[innerIter] * cos(angle);
        }

        // save sums to respective output element
        outRealNums[outerIter] = sumReal;
        outImagineNums[outerIter] = sumImagine;
    }
}


/*
 * 
 */
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Must provide single unsigned int" << std::endl;
        std::exit(0);
    }
    size_t N = std::stoul(argv[1]);

    
    // initialize the beginning arrays
    double inRealNums[N];
    double inImagineNums[N];
    double outRealNums[N];
    double outImagineNums[N];
    initialize(N, inRealNums, inImagineNums);
    
    // run DFT
    calculateDFT(N, inRealNums, inImagineNums, outRealNums, outImagineNums);
    
    std::cout << N 
            << std::endl
            << outRealNums[100] << " , " << outImagineNums[100]
            << std::endl 
            << outRealNums[39900] << " , " << outImagineNums[39900] 
            << std::endl;
    
    return 0;
}
