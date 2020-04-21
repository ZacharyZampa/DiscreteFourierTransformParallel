#include <cmath>
#include <iostream>
#include <mpi.h>

// Copyright 2020 Alan M. Ferrenberg

// Copyright (C) 2020 zampaze@miamioh.edu

using std::cerr;
using std::cout;
using std::endl;
using std::stoi;

#define ZERO 0.0000000000000000000000000000000000

/**
 * Compute the DFT - manager thread
 * @param N
 * @param oR
 * @param oI
 * @param fR
 * @param fI
 * @param numTasks
 */
void computeFTManagerLoop(int N, double oR[], double oI[], double fR[],
        double fI[], int numTasks) {
    int sentK, receivedK;
    MPI_Status status;

    // Loop over the N frequency values and send out to workers
    // loop ends when either out of N values 
    // or when out of numTasks (tasks) to send to
    for (sentK = 0; sentK < N && sentK < numTasks - 1; sentK++) {
        // Send k to a worker and have it
        MPI_Send(0, 0, MPI_DOUBLE, sentK + 1, sentK, MPI_COMM_WORLD);
    }

    // Receive the calculated DFT values from the workers
    // If sentK < N then send more values to the task workers
    for (receivedK = 0; receivedK < N; receivedK++) {
        double array[2];
        MPI_Recv(array, 2, MPI_DOUBLE, MPI_ANY_SOURCE,
                MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        // Place tempR, and tempI into their 
        // index (k value) in out arrays     
        fR[status.MPI_TAG] = array[0];
        fI[status.MPI_TAG] = array[1];

        // If sentK < N then send another value back to the task worker
        if (sentK < N) {
            // more K values
            MPI_Send(0, 0, MPI_DOUBLE, status.MPI_SOURCE,
                    sentK, MPI_COMM_WORLD);
            sentK++;
        } else {
            // work is all done, notify this task to shut down
            MPI_Send(MPI_BOTTOM, 0, MPI_DOUBLE, status.MPI_SOURCE,
                    sentK, MPI_COMM_WORLD);
        }
    }
}

/**
 * Compute the DFT - worker thread
 * @param N
 * @param oR
 * @param oI
 */
void computeFTWorkerLoop(int N, double oR[], double oI[]) {
    // loop worker for all values of 
    int k = 0;
    double tempR, tempI;
    MPI_Status status;
    
    // Receive k value from manager and begin computation.
    MPI_Recv(0, 0, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    k = status.MPI_TAG;  // snag the tag from the sent message

    while (k < N) {
        tempR = ZERO;
        tempI = ZERO;

        // Loop over the N spatial/temporal values
        for (int n = 0; n < N; n++) {
            double arg = 2.0 * M_PI * k * n / N;
            double cosArg = cos(arg);
            double sinArg = sin(arg);

            // Accumulate the real and imaginary components 
            // of the Fourier transform for frequency k in temporary variables
            tempR += oR[n] * cosArg + oI[n] * sinArg;
            tempI += oI[n] * cosArg - oR[n] * sinArg;
        }

        // Send what k, tempR, and tempI were sent back  
        double array[] = {tempR, tempI};
        MPI_Send(array, 2, MPI_DOUBLE, 0, k, MPI_COMM_WORLD);
        MPI_Recv(0, 0, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        k = status.MPI_TAG;
    }
}


// Initialize the real and imaginary components of the original function and
// the Fourier transform.  The function is sinc(x) = sin(ax)/x

void initialize(int N, double oR[], double oI[], double fR[], double fI[]) {
    double a = 2.0;

    oR[0] = a;
    oI[0] = ZERO;
    fR[0] = ZERO;
    fI[0] = ZERO;

    for (int n = 1; n < N; n++) {
        oR[n] = sin(a * n) / n;
        oI[n] = ZERO;
        fR[n] = ZERO;
        fI[n] = ZERO;
    }
}

void doMPIStuff(int argc, char *argv[], int *numTasks, int *rank) {
    int value;

    // Initialize MPI for this program
    if ((value = MPI_Init(&argc, &argv)) != 0) {
        std::cerr << "Problem with MPI_Init" << std::endl;
        exit(value);
    }

    // Get the number of processes, exit if there aren't 2 or more 
    if ((value = MPI_Comm_size(MPI_COMM_WORLD, numTasks)) != 0) {
        std::cerr << "Problem with MPI_Comm_size" << std::endl;
        exit(value);
    }
    if (*numTasks < 2) {
        std::cerr << "Program is designed to run with multiple tasks"
                << std::endl;
        exit(3);
    }

    // Get my rank
    if ((value = MPI_Comm_rank(MPI_COMM_WORLD, rank)) != 0) {
        std::cerr << "Problem with MPI_Comm_rank" << std::endl;
        exit(value);
    }
}

void manager(int N, double oR[], double oI[], double fR[], double fI[],
        double *eTime, double *cTime, int numTasks) {
    double CPU, totalCPU, eStart, eEnd;
    clock_t t1;

    // Initialize the clocks for elapsed and CPU time
    eStart = MPI_Wtime();
    t1 = clock();

    // send oR and oI arrays to all workers
    // The real part of the function
    MPI_Bcast(oR, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // The imaginary part of the function
    MPI_Bcast(oI, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform the manager tasks
    computeFTManagerLoop(N, oR, oI, fR, fI, numTasks);

    // Get the end CPU time and use MPI_Reduce to accumulate it
    t1 = clock() - t1;
    CPU = static_cast<double> (t1) / static_cast<double> (CLOCKS_PER_SEC);
    MPI_Reduce(&CPU, &totalCPU, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Get the end elapsed time and determine the transfer rate
    eEnd = MPI_Wtime();
    *eTime = (eEnd - eStart);
    *cTime = totalCPU;
}

void worker(int N) {
    double CPU, totalCPU;
    clock_t t1;

    // Initialize the clock for CPU time
    t1 = clock();

    double oR[N], oI[N];

    // The real part of the function
    MPI_Bcast(oR, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // The imaginary part of the function
    MPI_Bcast(oI, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform the worker tasks
    computeFTWorkerLoop(N, oR, oI);

    // Get the end CPU time and use MPI_Reduce to accumulate it
    t1 = clock() - t1;
    CPU = static_cast<double> (t1) / static_cast<double> (CLOCKS_PER_SEC);
    MPI_Reduce(&CPU, &totalCPU, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}

void mpiPart(int argc, char** argv, double* oR, double* oI, double* fR,
        double* fI, int N, int& numTasks, int& rank) {
    doMPIStuff(argc, argv, &numTasks, &rank);

    // Do differentiated work based on the rank.  First for the manager task
    if (rank == 0) {
        double elapsedTime, cpuTime;
        manager(N, oR, oI, fR, fI, &elapsedTime, &cpuTime, numTasks);
        cerr << "elapsed time = " << elapsedTime;
        cerr << " CPU time = " << cpuTime << std::endl;

        // Write out the real and imaginary components of the Fourier transform
        for (int k = 0; k < N; k++) {
            cout << k << " " << fR[k] << " " << fI[k] << endl;
        }
    } else {
        // Now tasks for the other process.
        worker(N);
    }

    // Wrap up MPI and clear out memory
    MPI_Finalize();

    // Free up the memory on the heap
    delete [] oR;
    delete [] oI;
    delete [] fR;
    delete [] fI;
}

// Main method for this program

int main(int argc, char *argv[]) {
    // Declare pointers to the Function and fourier transform arrays
    double *oR, *oI, *fR, *fI;

    // Check whether the number of samples has been provided
    if (argc < 2) {
        cerr << endl << "Usage:  fourier_serial number_of_samples." << endl;
        exit(1);
    }

    // Number of samples is the command line argument
    int N = stoi(argv[1]), numTasks, rank;


    // Allocate arrays for the function and its Fourier transform.  This puts
    // them in the heap, not the stack.
    oR = new double[N];
    oI = new double[N];
    fR = new double[N];
    fI = new double[N];

    // Initialize the original function and its Fourier transform
    initialize(N, oR, oI, fR, fI);

    // If OUTPUT is defined, print out the original function and exit
#ifdef OUTPUT
    for (int n = 0; n < N; n++) {
        cout << n << " " << oR[n] << " " << oI[n] << endl;
    }
    exit(0);
#endif

    // Compute the Fourier transform of the function - through MPI
    mpiPart(argc, argv, oR, oI, fR, fI, N, numTasks, rank);

    // Exit!
    exit(0);
}

