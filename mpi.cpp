#include "common.h"
#include <mpi.h>
#include <vector>

// Put any static global variables here that you will use throughout the simulation.
using std::vector;
typedef std::vector<particle_t> bin_type;
typedef std::vector<int> bin_type_idx;

vector<bin_type_idx> particle_bins_idx;

int num_proc_x, num_proc_y;  // total number of processors along the x- and y- axes
int proc_x, proc_y;
double left_x, right_x, bottom_y, top_y;
int neighbors[8];
int num_neighbors = 0;
int nlocal;
int nghosts = 0;

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
	neighbors[p_sw] = ((proc_x != 0)            && (proc_y != 0)           ) ? ((proc_y-1)*num_proc_x + (proc_x-1)) : (NONE);
	neighbors[p_s ] = (                            (proc_y != 0)           ) ? ((proc_y-1)*num_proc_x + (proc_x  )) : (NONE);
	neighbors[p_se] = ((proc_x != num_proc_x-1) && (proc_y != 0)           ) ? ((proc_y-1)*num_proc_x + (proc_x+1)) : (NONE);
	neighbors[p_w ] = ((proc_x != 0)                                       ) ? ((proc_y  )*num_proc_x + (proc_x-1)) : (NONE);
	neighbors[p_e ] = ((proc_x != num_proc_x-1)                            ) ? ((proc_y  )*num_proc_x + (proc_x+1)) : (NONE);
	neighbors[p_nw] = ((proc_x != 0)            && (proc_y != num_proc_y-1)) ? ((proc_y+1)*num_proc_x + (proc_x-1)) : (NONE);
	neighbors[p_n ] = (                            (proc_y != num_proc_y-1)) ? ((proc_y+1)*num_proc_x + (proc_x  )) : (NONE);
	neighbors[p_ne] = ((proc_x != num_proc_x-1) && (proc_y != num_proc_y-1)) ? ((proc_y+1)*num_proc_x + (proc_x+1)) : (NONE);
    
	for(int i = 0 ; i < 8; ++i) {
		if(neighbors[i] != NONE) num_neighbors++;
	}

    //  allocate storage for local particles, ghost particles
    particle_t *local = (particle_t*) malloc( num_parts * sizeof(particle_t) );
	char *p_valid = (char*) malloc(num_parts * sizeof(char));
	
    int ghost_packet_length[8];
    particle_t* ghost_packet_particles[8]; // In order of SW, S, SE, W, E, NW, N, NE
    MPI_Request mpi_ghost_requests[8];
    for(int i = 0; i < 8; ++i) {
		ghost_packet_particles[i] = (particle_t *) malloc(num_parts * sizeof(particle_t));
	}


	init_emigrant_buf(num_parts);

	nlocal = select_particles(num_parts, parts, local, p_valid, left_x, right_x, bottom_y, top_y);
	
	particle_t* ghost_particles = (particle_t *) malloc(num_parts * sizeof(particle_t));
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