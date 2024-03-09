#include "common.h"
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>

using std::vector;

#define GHOST_LENGTH (cutoff*2)
#define VALID   1
#define INVALID 0
#define EMIGRANT_SIZE_TAG    0
#define EMIGRANT_TAG         1
#define GHOST_SIZE_TAG       2
#define GHOST_TAG            3
const int NONE = -1;

// Put any static global variables here that you will use throughout the simulation.
double bottom_y, top_y;
int neighbors[2];
int num_neighbors = 0;

// Particles in this processor box
std::vector<particle_t> local_parts;    
int nlocal; // = local_parts.size();


/* For calculating forces */
// particles in this processor, who are in the margin region of the box (still inside). 
// In order of SW, S, SE, W, E, NW, N, NE
std::vector<particle_t> ghosts_outgoing[2];   
int ghosts_outgoing_size[2];   //outgoing ghost packet length in each direction. Previously called ghost_packet_length[8]   // ghosts_outgoing_size[i] = ghosts_outgoing[i].size(); 
std::vector<particle_t> ghosts_incomingBuffer[2];   // Incoming buffer for the ghosts
int ghosts_incomingBuffer_size[2];   // This is important for MPI_Recv  
MPI_Request mpi_ghost_requests[2];
std::vector<particle_t> all_incoming_ghosts;

/* For moving particles */
std::vector<particle_t> moving_out_parts[2];
int moving_out_size[2];
std::vector<particle_t> moving_in_parts[2];
int moving_in_size[2]; // This is important for MPI_Recv
MPI_Request mpi_move_requests[2];


void apply_force( particle_t &particle, particle_t &neighbor ) {

    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if( r2 > cutoff*cutoff )
        return;
    r2 = fmax( r2, min_r*min_r );
    double r = sqrt( r2 );

    //
    //  very simple short-range repulsive force
    //
    double coef = ( 1 - cutoff / r ) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}


void move( particle_t &p, double size )
{
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x  += p.vx * dt;
    p.y  += p.vy * dt;

    //
    //  bounce from walls
    //
    while( p.x < 0 || p.x > size )
    {
        p.x  = p.x < 0 ? -p.x : 2*size-p.x;
        p.vx = -p.vx;
    }
    while( p.y < 0 || p.y > size )
    {
        p.y  = p.y < 0 ? -p.y : 2*size-p.y;
        p.vy = -p.vy;
    }
}

// Only compute on local particles
void compute_forces(std::vector<particle_t>& in_the_box, std::vector<particle_t>& recvd_ghosts) {
	for(int i = 0; i < in_the_box.size(); ++i) {
        // printf("ComputeForces: num_particles, i, local[i].x = %d, %d, %f\n", num_particles, i, local[i].x); 
		
		in_the_box[i].ax = in_the_box[i].ay = 0;
		for (int j = 0; j < in_the_box.size(); ++j) {
			apply_force( in_the_box[i], in_the_box[j] );
		}
		
		for(int j = 0; j < recvd_ghosts.size(); ++j) {
			apply_force( in_the_box[i], recvd_ghosts[j]);
		}
	}
}


bool compare_particles(particle_t left, particle_t right) {
	return left.id < right.id;
}


void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	
	// Determine my cell boundaries
	bottom_y = (rank==0)            ? (0)    : ((size/num_procs)*rank);
	top_y    = (rank==num_proc_y-1) ? (size) : ((size/num_procs)*(rank+1));

	// Determine the ranks of my neighbors for message passing, NONE means no neighbor
    neighbors[0] = (rank != 0) ? (rank - 1) : (NONE);
    neighbors[1] = (rank == num_procs - 1) ? (rank + 1) : (NONE);
    
	for(int i = 0 ; i < 2; ++i) {
		if(neighbors[i] != NONE) num_neighbors++;
	}

    printf("neighbors[0] through [1]: %d, %d\n", neighbors[0],neighbors[1]);
    printf("num_neighbors = %d\n", num_neighbors);

	MPI_Barrier(MPI_COMM_WORLD);
    local_parts.clear(); 
    /* Note that this copies the particles by value. And thus, if local_parts[i].x is changed, it's not reflected in parts[i].x*/
    for(int i = 0; i < num_parts; ++i) {
		if((parts[i].x >= left_x) && (parts[i].x < right_x) && (parts[i].y >= bottom_y) && (parts[i].y < top_y)){
			local_parts.push_back(parts[i]);
		}
	}
	nlocal = local_parts.size();
    printf("nlocal = %d\n", nlocal); 
    printf("Done with INIT\n");
}


void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    all_incoming_ghosts.clear();
    for (int i = 0 ; i < 2; ++i) {
        ghosts_outgoing[i].clear();  // Reset so will just overwrite old packet data
        ghosts_incomingBuffer[i].clear();
        moving_out_parts[i].clear();
        moving_in_parts[i].clear();
    }

    // Handle ghosting for computing forces
    // Prepare ghost packets
	for(int i = 0; i < nlocal; ++i) {
		if(local_parts[i].y <= (bottom_y + GHOST_LENGTH)) { // check if in S ghost zone by y
			if((neighbors[0] != NONE)) { // Add to packet if S neighbor exists
                ghosts_outgoing[0].push_back(local_parts[i]);
			}
		}
		else if(local_parts[i].y >= (top_y - GHOST_LENGTH)) {// check if in N ghost zone by y
			if((neighbors[1] != NONE)) { // Add to packet if N neighbor exists
                ghosts_outgoing[1].push_back(local_parts[i]);
			}
		}
	}
    printf("Done with preparing ghost package. \n");

    // send/receive ghost packets size
    MPI_Request mpi_ghost_size_requests[2];
    MPI_Request mpi_ghost_incoming_size_requests[2];

    for(int i = 0; i < 2; ++i) {
        if (neighbors[i] != NONE) {
            ghosts_outgoing_size[i] = ghosts_outgoing[i].size();
            MPI_Isend(&(ghosts_outgoing_size[i]), 1, MPI_INT, neighbors[i], GHOST_SIZE_TAG, MPI_COMM_WORLD, &(mpi_ghost_size_requests[i]));
            MPI_Irecv(&(ghosts_incomingBuffer_size[i]), 1, MPI_INT, neighbors[i], GHOST_SIZE_TAG, MPI_COMM_WORLD, &(mpi_ghost_incoming_size_requests[i]));
        }
    }

    for (int i = 0; i < 2; ++i) {
        if (neighbors[i] != NONE) {
            MPI_wait(&(mpi_ghost_size_requests[i]))
        }
    }
    printf("Done with sending ghost package size. \n");

    for (int i = 0; i < 2; ++i) {
        if (neighbors[i] != NONE) {
            MPI_wait(&(mpi_ghost_incoming_size_requests[i]))
        }
    }
    printf("Done with receiving ghost package size. \n");

    // send/receive ghost packets data
    for(int i = 0; i < 2; ++i) {
        if (neighbors[i] != NONE) {
            ghosts_outgoing_size[i] = ghosts_outgoing[i].size();
            ghosts_incomingBuffer[i].resize(ghosts_incomingBuffer_size[i]);
            MPI_Isend(ghosts_outgoing[i].data(), ghosts_outgoing[i].size(), PARTICLE, neighbors[i],
                      GHOST_TAG, MPI_COMM_WORLD, &(mpi_ghost_data_requests[i]));
            MPI_Irecv(ghosts_incomingBuffer[i].data(), ghosts_incomingBuffer_size[i], PARTICLE,
                      neighbors[i], GHOST_TAG, MPI_COMM_WORLD, &(mpi_ghost_data_requests[i]));
        }
    }
    

    for (int i = 0; i < 2; ++i) {
        all_incoming_ghosts.insert(all_incoming_ghosts.end(), ghosts_incomingBuffer[i].begin(), ghosts_incomingBuffer[i].end());
    }

    
    //  Compute all forces
    compute_forces(local_parts, all_incoming_ghosts);
    printf("Done with computing forces. \n");
    
    //  Move particles
    for(int i = 0; i < local_parts.size(); ++i) {
        move( local_parts[i], size);
    }
    printf("Done with moving particles. \n");
    
    //  Handle particle migration
    printf("before emigration, nlocal = %d\n", local_parts.size()); 
    int exit_dir;
    auto iter = local_parts.begin();
    while (iter != local_parts.end()) {
        exit_dir = -1;
        if( (iter->y > top_y) && (neighbors[1] != -1)) {
            exit_dir = 1;    // moved north
        } else if( (iter->y < bottom_y) && (neighbors[0] != -1)) {
            exit_dir = 0;    // moved south
        } 

        if (exit_dir != -1) {
            moving_out_parts[exit_dir].push_back(*iter);
            /* vvv potential problem vvv */
            iter = local_parts.erase(iter); // Remove the current element and advance the iterator 
        } else {
            ++iter; // Move to the next element
        }
    }
    printf("After emigration, nlocal = %d\n", local_parts.size());

    // Send emigrants
    MPI_Request mpi_emigrant_out_size_requests[2];
    MPI_Request mpi_emigrant_in_size_requests[2];
    for(int i = 0; i < 2; ++i) {
		if(neighbors[i] != NONE) {
            moving_out_size[i] = moving_out_parts[i].size(); 
            MPI_Isend(&(moving_out_size[i]), 1, MPI_INT, neighbors[i], EMIGRANT_SIZE_TAG, MPI_COMM_WORLD, &(mpi_emigrant_out_size_requests[i]));
            MPI_Irecv(&(moving_in_size[i]), 1, MPI_INT, neighbors[i], EMIGRANT_SIZE_TAG, MPI_COMM_WORLD, &(mpi_emigrant_in_size_requests[i]))
		}
	}


            MPI_Isend(moving_in_parts[i].data(), moving_in_parts[i].size(), PARTICLE, neighbors[i], EMIGRANT_TAG, MPI_COMM_WORLD, &(mpi_move_requests[i]));
            MPI_Wait(&(mpi_move_requests[i]), MPI_STATUS_IGNORE);
    printf("Done with sending emigrant particles. \n");

    // receive immigrant particles
    MPI_Status status_migrant_size;
    MPI_Status status_migrant_data;
    for(int i = 0; i < 8; i++) {
        if(neighbors[i] == NONE) continue;

        MPI_Recv(&(moving_in_size[i]), 1, MPI_INT, neighbors[i], EMIGRANT_SIZE_TAG, MPI_COMM_WORLD, &status_migrant_size);

        moving_in_parts[i].resize(moving_in_size[i]);
        MPI_Recv(moving_in_parts[i].data(), moving_in_size[i], PARTICLE, neighbors[i], EMIGRANT_TAG, MPI_COMM_WORLD, &status_migrant_data);
    }
    // MPI_Waitall(num_neighbors, mpi_move_requests, MPI_STATUSES_IGNORE);
    printf("Done with receiving immigrant particles. \n");

    for (int i = 0; i < 8; ++i) {
        local_parts.insert(local_parts.end(), moving_in_parts[i].begin(), moving_in_parts[i].end());
    }
    nlocal = local_parts.size(); 


    /*
    printf("This is rank %d. There are %d local particles.\nThe positions are: \n", rank, nlocal);
	for(int i = 0; i < nlocal; ++i) {
		printf("%f %f    ", local_parts[i].x, local_parts[i].y);
	}
    */
    printf("\n");

}


void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.

    nlocal = local_parts.size();

    // Gather sizes of local_parts from all processors onto processor 0
    std::vector<int> allSizes(num_procs);
    MPI_Gather(&nlocal, 1, MPI_INT, allSizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute the total size of the concatenated vector on processor 0
    int total_size = 0;
    if (rank == 0) {
        for (int i = 0; i < num_procs; ++i) {
            total_size += allSizes[i];
        }
    }

    // Create a buffer to hold the concatenated local_parts on processor 0
    std::vector<particle_t> allParts(total_size);

    // Gather local_parts from all processors onto processor 0
    MPI_Gatherv(local_parts.data(), nlocal, PARTICLE, allParts.data(), allSizes.data(), allSizes.data(), PARTICLE, 0, MPI_COMM_WORLD);

	// Finally, sort the particles at node 0, and write to "parts" where we want the final answer
    if (rank == 0) {
        std::sort(allParts.begin(), allParts.end(), compare_particles);
        for (int i = 0; i < total_size; ++i) {
            parts[i] = allParts[i];
        }
    }
}

