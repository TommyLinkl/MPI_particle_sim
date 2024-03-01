#include "common.h"
#include <mpi.h>
#include <vector>

// Put any static global variables here that you will use throughout the simulation.
int navg, nabsavg=0;
double dmin, absmin=1.0,davg,absavg=0.0;
double rdavg,rdmin;
int rnavg;
int nlocal; 

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

    //  set up the data partitioning across processors
    int particle_per_proc = (num_parts + num_procs - 1) / num_procs;
    int *partition_offsets = (int*) malloc( (num_procs+1) * sizeof(int) );
    for( int i = 0; i < num_procs+1; i++ )
        partition_offsets[i] = min( i * particle_per_proc, num_parts );

    int *partition_sizes = (int*) malloc( num_procs * sizeof(int) );
    for( int i = 0; i < num_procs; i++ )
        partition_sizes[i] = partition_offsets[i+1] - partition_offsets[i];

    //  allocate storage for local partition
    nlocal = partition_sizes[rank];
    particle_t *local = (particle_t*) malloc( nlocal * sizeof(particle_t) );

    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    set_size( n );
    if( rank == 0 )
        init_particles( n, particles );
    MPI_Scatterv( particles, partition_sizes, partition_offsets, PARTICLE, local, nlocal, PARTICLE, 0, MPI_COMM_WORLD );
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    navg = 0;
    dmin = 1.0;
    davg = 0.0;

    //  collect all global data locally
    MPI_Allgatherv( local, nlocal, PARTICLE, particles, partition_sizes, partition_offsets, PARTICLE, MPI_COMM_WORLD );

    //  compute all forces
    for( int i = 0; i < nlocal; i++ ) {
        local[i].ax = local[i].ay = 0;
        for (int j = 0; j < n; j++ )
            apply_force( local[i], particles[j], &dmin, &davg, &navg );
    }

    if( find_option( argc, argv, "-no" ) == -1 )
    {

        MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);

        if (rank == 0){
            if (rnavg) {
                absavg +=  rdavg/rnavg;
                nabsavg++;
            }
            if (rdmin < absmin) absmin = rdmin;
        }
    }

    //  move particles
    for( int i = 0; i < nlocal; i++ )
        move( local[i] );
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