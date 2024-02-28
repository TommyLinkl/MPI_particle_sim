#include "common.h"
#include <mpi.h>
#include <vector>

using std::vector;
typedef std::vector<particle_t> bin_type;
typedef std::vector<int> bin_type_idx;

vector<bin_type_idx> particle_bins_idx;

double gridSize; 
double binSize; 
int binNum; 

// Put any static global variables here that you will use throughout the simulation.

void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}


void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}


void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    gridSize = sqrt(num_parts*density);
    binSize = cutoff * 2;  
    binNum = int(gridSize / binSize)+1; // Should be around sqrt(N/2)
    particle_bins_idx.resize(binNum * binNum); 

    for (int i = 0; i < num_parts; i++) {
        int x = int(parts[i].x / binSize);
        int y = int(parts[i].y / binSize);
        particle_bins_idx[x*binNum + y].push_back(i); 
    }

    // delete[] parts;
    // parts = NULL;
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // navg = 0;
    // dmin = 1.0;
    // davg = 0.0;
    
    //  compute forces
    for (int i = 0; i < binNum; i++) {
        for (int j = 0; j < binNum; j++) {            
            bin_type_idx& vec_idx = particle_bins_idx[i*binNum+j]; 
            for (int k = 0; k < vec_idx.size(); k++) {
                parts[vec_idx[k]].ax = parts[vec_idx[k]].ay = 0; 
            }
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {     // These for loops looks over a three by three stencil around the bin
                    if (i + dx >= 0 && i + dx < binNum && j + dy >= 0 && j + dy < binNum) {  // Edge cases
                        bin_type_idx& vec2_idx = particle_bins_idx[(i+dx) * binNum + j + dy]; 
                        for (int k = 0; k < vec_idx.size(); k++)
                            for (int l = 0; l < vec2_idx.size(); l++) {
                                apply_force(parts[vec_idx[k]], parts[vec2_idx[l]]);
                            }
                    }
                }
            }
        }
    }
    // printf("Done with compute force. \n");

    // Move particles
    bin_type_idx temp;
    for (int i = 0; i < binNum; i++) {
        for(int j = 0; j < binNum; j++) {            
            bin_type_idx& vec_idx = particle_bins_idx[i * binNum + j]; 
            int tail = vec_idx.size(), k = 0;
            while(k < tail) {
                move( parts[vec_idx[k]], size); 
                int x = int(parts[vec_idx[k]].x / binSize); //Check the position
                int y = int(parts[vec_idx[k]].y / binSize);
                if (x == i && y == j)  // Still inside original bin
                    k++;
                else
                {
                    temp.push_back(vec_idx[k]);  // Store paricles that have changed bin. 
                    vec_idx[k] = vec_idx[--tail]; //Remove it from the current bin.
                }
            }
            vec_idx.resize(k);
        }
    }
    // printf("Done with moving particles. \n");

    // Rebin the particles
    for (int i = 0; i < temp.size(); ++i) {
            int x = temp[i].x / binSize;
            int y = temp[i].y / binSize;
            bins[x*binNum + y].push_back(temp[i]);
        }

    //  Deleting the temp list
    temp.clear();
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.

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