# DiscreteFourierTransformParallel
Three methods of for parallelization  of the discrete fourier transform.

fourier_mpi allows for MPI parallelization. This parallelization is accomplished by utilizing the manager / worker model.
fourier_omp_inner and fourier_omp_outer uses OMP parallelization. fourier_omp_inner parallelizes the inner loop, fourier_omp_outer parallelizes the outer loop
