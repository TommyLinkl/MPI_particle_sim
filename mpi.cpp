#include "common.h"
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>

#define GHOST_LENGTH (cutoff*2)
#define VALID   1
#define INVALID 0
#define EMIGRANT_TAG    1
#define GHOST_TAG       2
const int NONE = -1;

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
char *p_valid; 
particle_t *local; 
particle_t *ghost_particles; 
particle_t **emigrant_buf;
particle_t *immigrant_buf;
int *emigrant_cnt;
MPI_Request mpi_em_requests[8];

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


void compute_forces(particle_t local[], char p_valid[], int num_particles, particle_t ghosts[], int num_ghosts) {
	int seen_particles = 0;
	for(int i = 0; seen_particles < num_particles; ++i) {
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


int add_particle(particle_t &new_particle, int array_sz, particle_t *particles, char* p_valid)
{
    int insert_idx = -1;

    // Find a new slot for the particle
    for(int i = 0; i < array_sz; i++)
    {
        if(p_valid[i] == INVALID)
        {
            particles[i] = new_particle;
            p_valid[i] = VALID;
            insert_idx = i;
            break;
        }
    }

    return insert_idx;
}


bool compare_particles(particle_t left, particle_t right) {
	return left.id < right.id;
}


void receive_ghost_packets(int* num_ghost_particles, particle_t* ghost_particles, int* neighbors, int num_neighbors, int buf_size)
{
    MPI_Status status;
	
	*num_ghost_particles = 0;

    for(int i = 0; i < 8; i++)
    {
		int num_particles_rcvd = 0;
		
        // If no neighbor in this direction, skip over it
        if(neighbors[i] == NONE) continue;

        // Perform blocking read from neighbor
        MPI_Recv (ghost_particles+(*num_ghost_particles), (buf_size-(*num_ghost_particles)), PARTICLE, neighbors[i], GHOST_TAG, MPI_COMM_WORLD, &status); 

        MPI_Get_count(&status, PARTICLE, &num_particles_rcvd);
		*num_ghost_particles += num_particles_rcvd;
    }

    // Make sure that all previous emigrant messages have been sent, as we need to reuse the buffers
    MPI_Waitall(num_neighbors, mpi_ghost_requests, MPI_STATUSES_IGNORE);
}


void prepare_emigrants(particle_t* particles, char* p_valid, int* num_particles, double left_x, double right_x, double bottom_y, double top_y, int* neighbors)
{
    int num_particles_checked = 0;
    int num_particles_removed = 0;
    int exit_dir;

    // Initialize all emigrant counts to zero
    for(int i = 0; i < 8; i++)
        emigrant_cnt[i] = 0;

    // Loop through all the particles, checking if they have left the bounds of this processor
    for(int i = 0; num_particles_checked < (*num_particles); i++) {
        if(p_valid[i] == INVALID)
            continue;

        exit_dir = -1;

        // If moved north-west
        if( (particles[i].y > top_y) && (particles[i].x < left_x) && (neighbors[5] != -1))
        {
            exit_dir = 5;
        }

        // Else if moved north-east
        else if( (particles[i].y > top_y) && (particles[i].x > right_x) && (neighbors[7] != -1))
        {
            exit_dir = 7;
        }

        // Else if moved north
        else if( (particles[i].y > top_y) && (neighbors[6] != -1))
        {
            exit_dir = 6;
        }

        // Else if moved south-west
        else if( (particles[i].y < bottom_y) && (particles[i].x < left_x) && (neighbors[0] != -1))
        {
            exit_dir = 0;
        }

        // Else if moved south-east
        else if( (particles[i].y < bottom_y) && (particles[i].x > right_x) && (neighbors[2] != -1))
        {
            exit_dir = 2;
        }

        // Else if moved south
        else if( (particles[i].y < bottom_y) && (neighbors[1] != -1))
        {
            exit_dir = 1;
        }

        // Else if moved west
        else if( (particles[i].x < left_x) && (neighbors[3] != -1))
        {
            exit_dir = 3;
        }

        // Else if moved east
        else if( (particles[i].x > right_x) && (neighbors[4] != -1))
        {
            exit_dir = 4;
        }

        // If the particle is an emigrant, remove it from the array and place it in the correct buffer
        if(exit_dir != -1) {
            emigrant_buf[exit_dir][emigrant_cnt[exit_dir]] = particles[i];
            emigrant_cnt[exit_dir] += 1;
            p_valid[i] = INVALID;
            num_particles_removed++;
        }

        num_particles_checked++;
    }

    // Update the count of active particles on this processor
    (*num_particles) -= num_particles_removed;
}


//
// Actually sends the immigrants from this processor to neighboring processors.
// The emigrants are determined in the 'prepare_emigrants' function. This function sends the 
// contents of the global emigrant_buf arrays to the neighbors.
// 'neighbors' is indexed by the direction values (e.g. N=0, S=1, etc) and gives the corresponding neighbor's rank.
// If a neighbor is not present, its direction is indicated as -1.
//
void send_emigrants(int* neighbors)
{
    int num_requests = 0;

	for(int i = 0; i < 8; ++i) {
		if(neighbors[i] != NONE) {
            MPI_Isend ((void*)(emigrant_buf[i]), emigrant_cnt[i], PARTICLE, neighbors[i], EMIGRANT_TAG, MPI_COMM_WORLD, &(mpi_em_requests[num_requests]));
            num_requests++;
		}
	}
}


//
// Receives immigrant particles from neighboring processors, and adds
// them to the list of local particles.
// 'neighbors' is indexed by the direction values (e.g. N=0, S=1, etc) and gives the corresponding neighbor's rank.
// If a neighbor is not present, its direction is indicated as -1.
// 'num_particles' is the number of valid particles on this processor, and is updated once all particles have been received.
// 'buf_size' is in terms of number of particles, not bytes
//
void receive_immigrants(int* neighbors, int num_neighbors, particle_t* particles, char* p_valid, int* num_particles, int array_sz, int buf_size)
{
    MPI_Status status;
    int num_particles_rcvd = 0;

    for(int i = 0; i < 8; i++) {
        // If no neighbor in this direction, skip over it
        if(neighbors[i] == -1)
            continue;

        // Perform blocking read from neighbor
        MPI_Recv ((void*)(immigrant_buf), buf_size, PARTICLE, neighbors[i], EMIGRANT_TAG, MPI_COMM_WORLD, &status); 

        MPI_Get_count(&status, PARTICLE, &num_particles_rcvd);

        // If the neighbor sent particles, add them to the local particle list
        for(int j = 0; j < num_particles_rcvd; j++) {
            if(add_particle(immigrant_buf[j], array_sz, particles, p_valid) == -1) {
                printf("Error: insufficient space to add particle to local array\n");
                exit(-1);
            }
        }

        // Update the number of particles on the local processor
        (*num_particles) += num_particles_rcvd;
    }

    // Make sure that all previous emigrant messages have been sent, as we need to reuse the buffers
    MPI_Waitall(num_neighbors, mpi_em_requests, MPI_STATUSES_IGNORE);
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
	left_x   = (proc_x==0)            ? (0)        : ((size/num_proc_x)*proc_x);
	right_x  = (proc_x==num_proc_x-1) ? (size) : ((size/num_proc_x)*(proc_x+1));
	bottom_y = (proc_y==0)            ? (0)        : ((size/num_proc_y)*proc_y);
	top_y    = (proc_y==num_proc_y-1) ? (size) : ((size/num_proc_y)*(proc_y+1));

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
    local = (particle_t*) malloc( num_parts * sizeof(particle_t) );
	ghost_particles = (particle_t *) malloc(num_parts * sizeof(particle_t));
	p_valid = (char*) malloc(num_parts * sizeof(char));   // Tells me which particles are in this box and in its ghost zone
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
    printf("Done with INIT");
}


void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    
    //  Handle ghosting

    // prepare ghost packets
    for (int i = 0 ; i < 8; ++i) {
        ghost_packet_length[i] = 0;  // Reset so will just overwrite old packet data
    }
    int seen_particles = 0;
	for(int i = 0; seen_particles < num_parts; ++i) {
		if(p_valid[i] == INVALID) continue;
		seen_particles++;
		
		if(parts[i].x <= (left_x + GHOST_LENGTH)) { // check if in W, SW, or NW ghost zone by x
			if((neighbors[3] != -1)) { // Add to packet if W neighbor exists
				ghost_packet_particles[3][ghost_packet_length[3]] = parts[i];
				++ghost_packet_length[3];
			}
			if(parts[i].y <= (bottom_y + GHOST_LENGTH)) { // Add to packet if SW neighbor exists and y bounded
				if((neighbors[0] != -1)) {
					ghost_packet_particles[0][ghost_packet_length[0]] = parts[i];
					++ghost_packet_length[0];
				}
			}
			else if (parts[i].y >= (top_y - GHOST_LENGTH)) { // Add to packet if NW neighbor exists and y bounded
				if((neighbors[5] != -1)) {
					ghost_packet_particles[5][ghost_packet_length[5]] = parts[i];
					++ghost_packet_length[5];
				}
			}
		}
		else if(parts[i].x >= (right_x - GHOST_LENGTH)) { // check if in E, SE, or NE ghost zone by x
			if((neighbors[4] != -1)) { // Add to packet if E neighbor exists
				ghost_packet_particles[4][ghost_packet_length[4]] = parts[i];
				++ghost_packet_length[4];
			}
			if(parts[i].y <= (bottom_y + GHOST_LENGTH)) { // Add to packet if SE neighbor exists and y bounded
				if((neighbors[2] != -1)) {
					ghost_packet_particles[2][ghost_packet_length[2]] = parts[i];
					++ghost_packet_length[2];
				}
			}
			else if (parts[i].y >= (top_y - GHOST_LENGTH)) { // Add to packet if NE neighbor exists and y bounded
				if((neighbors[7] != -1)) {
					ghost_packet_particles[7][ghost_packet_length[7]] = parts[i];
					++ghost_packet_length[7];
				}
			}
		}
		
		if(parts[i].y <= (bottom_y + GHOST_LENGTH)) { // check if in S ghost zone by y (SW, SE already handled)
			if((neighbors[1] != -1)) { // Add to packet if S neighbor exists
				ghost_packet_particles[1][ghost_packet_length[1]] = parts[i];
				++ghost_packet_length[1];
			}
		}
		else if(parts[i].y >= (top_y - GHOST_LENGTH)) {// check if in N ghost zone by y (NW, NE already handled)
			if((neighbors[6] != -1)) {
				ghost_packet_particles[6][ghost_packet_length[6]] = parts[i];
				++ghost_packet_length[6];
			}
		}
	}
    printf("Done with preparing ghost package. ");

    // send ghost packets
    int rc = 0;
	for(int i = 0; i < 8; ++i) {
		if(neighbors[i] != NONE) {
			MPI_Isend(ghost_packet_particles[i], ghost_packet_length[i], PARTICLE, neighbors[i], GHOST_TAG, MPI_COMM_WORLD, &(mpi_ghost_requests[rc++]));
		}
	}
    printf("Done with sending ghost package. ");

    // receive ghost packets
    receive_ghost_packets(&nghosts, ghost_particles, neighbors, num_neighbors, num_parts);
    printf("Done with receiving ghost package. ");
    
    //  Compute all forces
    compute_forces(local, p_valid, nlocal, ghost_particles, nghosts);
    printf("Done with computing forces. ");
    
    //  Move particles
    seen_particles = 0;
    for(int i = 0; seen_particles < nlocal; ++i) {
        if(p_valid[i] == INVALID) continue;
        seen_particles++;
        move( local[i], size);
    }
    printf("Done with moving particles. ");
    
    //  Handle migration
    prepare_emigrants(local, p_valid, &nlocal, left_x, right_x, bottom_y, top_y, neighbors);
    send_emigrants(neighbors);
    receive_immigrants(neighbors, num_neighbors, local, p_valid, &nlocal, num_parts, num_parts);

}


void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    // prepare_save(rank, n_proc, local, p_valid, nlocal, particles, n);

    // First, get the number of particles in each node into node 0. Also prepare array placement offsets.
	int* node_particles_num    = (int *) malloc(num_procs*sizeof(int));
	int* node_particles_offset = (int *) malloc(num_procs*sizeof(int));
	
	MPI_Gather(&nlocal, 1, MPI_INT, node_particles_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	if(rank == 0) {
		node_particles_offset[0] = 0;
		for(int i = 1; i < num_procs; ++i) {
			node_particles_offset[i] = node_particles_offset[i-1] + node_particles_num[i-1];
		}
	}
	
	// Now, each node prepares a collapsed list of all valid particles
	particle_t* collapsed_local = (particle_t *) malloc(nlocal * sizeof(particle_t));
	
	int seen_particles = 0;
	for(int i = 0; seen_particles < nlocal; ++i) {
		if(p_valid[i] == INVALID) continue;
		collapsed_local[seen_particles] = local[i];
		seen_particles++;
	}
	
	// Next, send the particles to node 0
	MPI_Gatherv(collapsed_local, nlocal, PARTICLE, parts, node_particles_num, node_particles_offset, PARTICLE, 0, MPI_COMM_WORLD);
	
	// Finally, sort the particles at node 0
	if(rank == 0) {
		std::sort(parts, parts + num_parts, compare_particles);
	}
	
	free(collapsed_local);
	free(node_particles_num);
	free(node_particles_offset);
}

