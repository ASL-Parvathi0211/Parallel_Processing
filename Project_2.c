#include <stdio.h>  // Standard input/output functions
#include <stdlib.h> // General-purpose standard library
#include <mpi.h>    // MPI for parallel computing
#include <math.h>   // Mathematical operations

#define SIGN(x) ((x) < 0.0 ? -1.0 : 1.0) // Utility macro to get the sign of a value
#define CONST1 1.23456                   // Constant used in the force calculation formula
#define CONST2 6.54321                   // Another constant used in the force calculation formula
#define ROOT_PROCESS 0                   // Identifier for the root MPI process

// MPI global variables
int process_rank, num_processes; // Rank of the current process and total number of processes

// Function to calculate forces between particles in a distributed manner
void calc_force(int num_particles, double *positions, double *forces) {
    int i, j;               // Loop indices for iterating over particles
    double distance, force; // Variables to store distance and computed force between particles

    // Initialize the forces array to zero for all particles
    for (i = 0; i < num_particles; i++) {
        forces[i] = 0.0;  // Set initial force to zero for each particle
    }

    // Calculate the portion of particles each process will handle
    int segment_size = num_particles / num_processes;  // Number of particles assigned to each process
    int start_index = process_rank * segment_size;     // Start index for the current process
    int end_index = (process_rank == num_processes - 1) ? num_particles : start_index + segment_size; // End index, ensuring last process takes remaining particles

    // Compute forces for particles within the assigned range
    for (i = start_index; i < end_index; i++) {
        for (j = 0; j < num_particles; j++) {
            if (i != j) {  // Skip force calculation for the particle interacting with itself
                distance = positions[i] - positions[j];  // Calculate the distance between particles i and j
                force = CONST1 / pow(distance, 3) - CONST2 * SIGN(distance) / (distance * distance);  // Apply the force calculation formula
                forces[i] += force;  // Accumulate the calculated force for particle i
            }
        }
    }

    // Use MPI to aggregate forces calculated by all processes
    MPI_Allreduce(MPI_IN_PLACE, forces, num_particles, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  // Perform an all-reduce to sum forces across all processes

    // Print results if this is the root process
    if (process_rank == ROOT_PROCESS) {
        for (i = 0; i < num_particles; i++) {
            printf("Particle %d: Force = %f\n", i, forces[i]);  // Display the final force acting on each particle
        }
    }
}

int main(int argc, char **argv) {
    int num_particles = 8;  // Total number of particles in the system
    double positions[8] = {5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2, 12.2};  // Array representing the positions of the particles
    double forces[8];  // Array to store the calculated forces for each particle

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);  // Initialize the MPI library
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);  // Get the rank (ID) of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes); // Get the total number of processes involved

    // Calculate the forces acting on each particle
    calc_force(num_particles, positions, forces);  // Invoke the function to calculate forces

    // Finalize the MPI environment
    MPI_Finalize();  // Clean up and shut down the MPI environment

    return 0;  // Exit the program successfully
}
