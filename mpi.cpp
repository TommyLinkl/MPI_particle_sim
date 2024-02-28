#include "common.h"
#include <mpi.h>

// Put any static global variables here that you will use throughout the simulation.

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here



    // 
    // Assgning equal bins to particles like the previous methods
    //
    
    vector<bin_type> bins;

    gridSize = sqrt(n * _density);
    binSize = _cutoff;  
    binNum = int(gridSize / binSize) + 1; 
    
    bins.resize(binNum * binNum);

    for (int i = 0; i < n; i++)
    {
        int x = int(particles[i].x / binSize);
        int y = int(particles[i].y / binSize);
        bins[x*binNum + y].push_back(particles[i]);
    }

    delete[] particles;
    particles = NULL;
    
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    navg = 0;
        dmin = 1.0;
        davg = 0.0;

         //
        //  compute forces
        //  Iterates over all the bins that contain the particle
        //  The apply force function will only be done on the local neighbourhood of the bin in which the particle belongs to
        //  The checking will be done, up, down, left, right and the corners
        //

        for (int i = 0; i < binNum; i++) {
            for (int j = 0; j < binNum; j++) {

                bin_type& vec = bins[i * binNum + j];
                
                for (int k = 0; k < vec.size(); k++)
                    vec[k].ax = vec[k].ay = 0;

                for (int dx = -1; dx <= 1; dx++)   // These for loops looks over a three by three stencil around the particle (left and right)
                {
                    for (int dy = -1; dy <= 1; dy++)  // up and down
                    {
                        if (i + dx >= 0 && i + dx < binNum && j + dy >= 0 && j + dy < binNum)    // Takes care of corners and wall
                        {
                            bin_type& vec2 = bins[(i+dx) * binNum + j + dy];
                            for (int k = 0; k < vec.size(); k++)
                                for (int l = 0; l < vec2.size(); l++)
                                    apply_force( vec[k], vec2[l], &dmin, &davg, &navg);      // executes the apply_force function from common.cpp
                        }
                    }
                }
            }
        }

        //
        //  move particles
        //

        bin_type temp;

        for (int i = 0; i < binNum; i++){
            for(int j = 0; j < binNum; j++)
            {
                bin_type& vec = bins[i * binNum + j];
                int tail = vec.size(), k = 0;
                //for(; k < tail; )
                while(k < tail)
                {
                    move( vec[k] );
                    int x = int(vec[k].x / binSize);  //Check the position
                    int y = int(vec[k].y / binSize);
                    if (x == i && y == j)  // Still inside original bin
                        k++;
                    else
                    {
                        temp.push_back(vec[k]);  // Store paricles that have changed bin. 
                        vec[k] = vec[--tail]; //Remove it from the current bin.
                    }
                }
                vec.resize(k);
            }
        }

        //
        //  Rebinning the particles
        //

        for (int i = 0; i < temp.size(); ++i) {
            int x = temp[i].x / binSize;
            int y = temp[i].y / binSize;
            bins[x*binNum + y].push_back(temp[i]);
        }

        //
        //  Deleting the temp list
        //

        temp.clear();

        if (find_option( argc, argv, "-no" ) == -1) {
          MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
          if (rank == 0){
            if (rnavg) {
              absavg +=  rdavg/rnavg;
              nabsavg++;
            }
            if (rdmin < absmin) 
                absmin = rdmin;
          }
        }
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
}