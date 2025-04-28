#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/mkl/2025.0/lib
export MAD_NUM_THREADS=8
./plot2cube file=alpha_orbital_0
./plot2cube file=alpha_orbital_1
./plot2cube file=alpha_orbital_2
./plot2cube file=alpha_orbital_3
./plot2cube file=beta_orbital_0
./plot2cube file=beta_orbital_1
./plot2cube file=beta_orbital_2
./plot2cube file=beta_orbital_3