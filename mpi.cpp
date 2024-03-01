#include "common.h"
#include <mpi.h>
#include <vector>

#define GHOST_LENGTH (cutoff*2)

// Put any static global variables here that you will use throughout the simulation.
int num_proc_x, num_proc_y;  // total number of processors along the x- and y- axes
int proc_x, proc_y;
double left_x, right_x, bottom_y, top_y;
int neighbors[8];
int num_neighbors = 0;
int nlocal;
int nghosts = 0;
int ghost_packet_length[8];
MPI_Request mpi_ghost_requests[8];
particle_t* ghost_packet_particles[8]; // In order of SW, S, SE, W, E, NW, N, NE

particle_t **emigrant_buf;
particle_t *immigrant_buf;
int *emigrant_cnt;
MPI_Request mpi_em_requests[8];

void compute_forces(particle_t local[], char p_valid[], int num_particles, particle_t ghosts[], int num_ghosts) {
	int seen_particles = 0;
	for(int i = 0; seen_particles < num_particles; ++i)
	{
		if(p_valid[i] == INVALID) continue;
		seen_particles++;
		
		local[i].ax = local[i].ay = 0;
		int nearby_seen_particles = 0;
		for (int j = 0; nearby_seen_particles < num_particles; ++j) {
			if(p_valid[j] == INVALID) continue;
			nearby_seen_particles++;
			
			apply_force( local[i], local[j] );
		}
		
		for(int j = 0; j < num_ghosts; ++j) {
			apply_force( local[i], ghosts[j]);
		}
	}
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
	for(int factor = (int)floor(sqrt((double)num_procs)); factor >= 1; --factor) {
		if(num_procs % factor == 0) {
			num_proc_x = factor;
			num_proc_y = num_procs/factor;
			break;
		}
	}
	
	// Determine where this cell is
	proc_x = rank % num_proc_x;
	proc_y = rank/num_proc_x;
	
	// Determine my cell boundaries
	left_x   = (proc_x==0)            ? (0)        : ((sim_size/num_proc_x)*proc_x);
	right_x  = (proc_x==num_proc_x-1) ? (sim_size) : ((sim_size/num_proc_x)*(proc_x+1));
	bottom_y = (proc_y==0)            ? (0)        : ((sim_size/num_proc_y)*proc_y);
	top_y    = (proc_y==num_proc_y-1) ? (sim_size) : ((sim_size/num_proc_y)*(proc_y+1));

	// Determine the ranks of my neighbors for message passing, NONE means no neighbor
	neighbors[0] = ((proc_x != 0)            && (proc_y != 0)           ) ? ((proc_y-1)*num_proc_x + (proc_x-1)) : (NONE);
	neighbors[1] = (                            (proc_y != 0)           ) ? ((proc_y-1)*num_proc_x + (proc_x  )) : (NONE);
	neighbors[2] = ((proc_x != num_proc_x-1) && (proc_y != 0)           ) ? ((proc_y-1)*num_proc_x + (proc_x+1)) : (NONE);
	neighbors[3] = ((proc_x != 0)                                       ) ? ((proc_y  )*num_proc_x + (proc_x-1)) : (NONE);
	neighbors[4] = ((proc_x != num_proc_x-1)                            ) ? ((proc_y  )*num_proc_x + (proc_x+1)) : (NONE);
	neighbors[5] = ((proc_x != 0)            && (proc_y != num_proc_y-1)) ? ((proc_y+1)*num_proc_x + (proc_x-1)) : (NONE);
	neighbors[6] = (                            (proc_y != num_proc_y-1)) ? ((proc_y+1)*num_proc_x + (proc_x  )) : (NONE);
	neighbors[7] = ((proc_x != num_proc_x-1) && (proc_y != num_proc_y-1)) ? ((proc_y+1)*num_proc_x + (proc_x+1)) : (NONE);
    
	for(int i = 0 ; i < 8; ++i) {
		if(neighbors[i] != NONE) num_neighbors++;
	}

    //  allocate storage for local particles, ghost particles
    particle_t *local = (particle_t*) malloc( num_parts * sizeof(particle_t) );
	particle_t* ghost_particles = (particle_t *) malloc(num_parts * sizeof(particle_t));
	char *p_valid = (char*) malloc(num_parts * sizeof(char));   // Tells me which particles are in this box
    for(int i = 0; i < 8; ++i) {   // In order of SW, S, SE, W, E, NW, N, NE
		ghost_packet_particles[i] = (particle_t *) malloc(num_parts * sizeof(particle_t));
	}

    immigrant_buf = (particle_t*)malloc(num_parts * sizeof(particle_t));   // Receive from neighbors
    emigrant_buf = (particle_t**)malloc(8 * sizeof(particle_t*));  // Send out to all 8 neighbors
    emigrant_cnt = (int*)malloc(num_parts * sizeof(int));
    for(int i = 0; i < 8; i++) {
        emigrant_buf[i] = (particle_t*)malloc(num_parts * sizeof(particle_t));
    }

    int current_particle = 0;
	for(int i = 0; i < num_parts; ++i) {
		if((parts[i].x >= left_x) && (parts[i].x < right_x) && (parts[i].y >= bottom_y) && (parts[i].y < top_y)){
			local[current_particle] = parts[i];
			p_valid[current_particle] = VALID;
			current_particle++;
		}
	}
	nlocal = current_particle; 

}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    //  Handle ghosting
    prepare_ghost_packets(local, p_valid, nlocal, left_x, right_x, bottom_y, top_y, neighbors);
    send_ghost_packets(neighbors);
    receive_ghost_packets(&nghosts, ghost_particles, neighbors, num_neighbors, n);
    
    //  Compute all forces
    compute_forces(local, p_valid, nlocal, ghost_particles, nghosts);
    
    //  Move particles
    int seen_particles = 0;
    
    for(int i = 0; seen_particles < nlocal; ++i)
    {
        if(p_valid[i] == INVALID) continue;
        seen_particles++;
        
        move( local[i] );
    }
    
    //
    //  Handle migration
    //
    prepare_emigrants(local, p_valid, &nlocal, left_x, right_x, bottom_y, top_y, neighbors);
    send_emigrants(neighbors);
    receive_immigrants(neighbors, num_neighbors, local, p_valid, &nlocal, n, n);
    
    //
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