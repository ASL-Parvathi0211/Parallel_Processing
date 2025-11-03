#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define FIND_MIN(a, b) ((a) < (b) ? (a) : (b)) // Macro to determine the smaller of two values

int main(int argc, char **argv)
{
    int rank, total_procs;                       // Rank identifier and process count for MPI
    MPI_Init(&argc, &argv);                      // Initialize MPI environment and process communication
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);        // Obtain the rank (ID) of each individual process
    MPI_Comm_size(MPI_COMM_WORLD, &total_procs); // Obtain the total number of processes active

    // Check that there are enough processes to proceed
    if (total_procs < 2)
    {
        if (rank == 0)
        {
            printf("Please run this program with at least 2 processes.\n"); // Warning if insufficient processes
        }
        MPI_Finalize(); // Exit MPI if minimum process count is not met
        return 0;       // End the program
    }

    int initialMatrix[8][8], resultMatrix[8][8], matrix_size = 8; // Define matrices of size 8x8 for initial and processed data

    // Step 1: Divide processes into groups based on even and odd ranks
    MPI_Group global_group, even_group, odd_group; // Groups for organizing processes by rank parity
    MPI_Comm even_comm, odd_comm;                  // Communicators to handle even and odd groups independently

    MPI_Comm_group(MPI_COMM_WORLD, &global_group); // Obtain the global group from MPI_COMM_WORLD

    // Arrays to hold process ranks for the even and odd groups
    int even_ranks[total_procs], odd_ranks[total_procs];
    int even_counter = 0, odd_counter = 0; // Counters for the number of ranks in each group

    // Assign process ranks to either even or odd groups
    for (int i = 0; i < total_procs; i++)
    {
        if (i % 2 == 0)
        {
            even_ranks[even_counter++] = i; // Add rank to even_ranks if it’s an even number
        }
        else
        {
            odd_ranks[odd_counter++] = i; // Add rank to odd_ranks if it’s an odd number
        }
    }

    // Generate the MPI groups and communicators for each set of ranks
    MPI_Group_incl(global_group, even_counter, even_ranks, &even_group); // Create the even group
    MPI_Group_incl(global_group, odd_counter, odd_ranks, &odd_group);    // Create the odd group

    MPI_Comm_create(MPI_COMM_WORLD, even_group, &even_comm); // Create communicator for the even group
    MPI_Comm_create(MPI_COMM_WORLD, odd_group, &odd_comm);   // Create communicator for the odd group

    // Step 2: Initialize the matrix values on the root process (rank 0)
    if (rank == 0)
    {
        int exampleMatrix[8][8] = {// Sample data for an 8x8 matrix
                                   {0, 2, 9, 3, 8, 5, 4, 7},
                                   {2, 0, 4, 1, 7, 6, 3, 5},
                                   {9, 4, 0, 5, 2, 9, 6, 8},
                                   {3, 1, 5, 0, 4, 8, 7, 9},
                                   {8, 7, 2, 4, 0, 3, 5, 6},
                                   {5, 6, 9, 8, 3, 0, 2, 4},
                                   {4, 3, 6, 7, 5, 2, 0, 1},
                                   {7, 5, 8, 9, 6, 4, 1, 0}};

        // Copy the exampleMatrix data into initialMatrix for processing
        for (int i = 0; i < matrix_size; i++)
        {
            for (int j = 0; j < matrix_size; j++)
            {
                initialMatrix[i][j] = exampleMatrix[i][j];
            }
        }

        // Print the initialized matrix for verification
        printf("Initial Matrix \n\n");
        for (int i = 0; i < matrix_size; i++)
        {
            for (int j = 0; j < matrix_size; j++)
            {
                printf("%4d ", initialMatrix[i][j]); // Print matrix values with formatted spacing
            }
            printf("\n");
        }
    }

    // Share the initial matrix data across all processes
    MPI_Bcast(initialMatrix, matrix_size * matrix_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Copy initialMatrix into resultMatrix for subsequent computations
    for (int i = 0; i < matrix_size; i++)
    {
        for (int j = 0; j < matrix_size; j++)
        {
            resultMatrix[i][j] = initialMatrix[i][j];
        }
    }

    // Step 3: Perform calculations using groups with separate communicators
    for (int k = 0; k < matrix_size; k++)
    {
        // Share the k-th row among processes in each communicator group
        if (rank % 2 == 0 && even_comm != MPI_COMM_NULL)
        {
            MPI_Bcast(&initialMatrix[k], matrix_size, MPI_INT, 0, even_comm); // Broadcast within the even group
        }
        else if (rank % 2 != 0 && odd_comm != MPI_COMM_NULL)
        {
            MPI_Bcast(&initialMatrix[k], matrix_size, MPI_INT, 0, odd_comm); // Broadcast within the odd group
        }

        // Define which rows each group will be responsible for updating
        int row_start = (rank % 2 == 0) ? 0 : 4; // Rows 0-3 for even group, 4-7 for odd group
        int row_end = row_start + 4;             // Each group handles four rows

        // Update the resultMatrix by comparing and finding the minimum values within assigned rows
        for (int i = row_start; i < row_end; i++)
        {
            for (int j = 0; j < matrix_size; j++)
            {
                // Apply the shortest path formula
                resultMatrix[i][j] = FIND_MIN(resultMatrix[i][j], resultMatrix[i][k] + initialMatrix[k][j]);
            }
        }

        // Consolidate and synchronize matrix values across processes
        MPI_Allreduce(MPI_IN_PLACE, resultMatrix, matrix_size * matrix_size, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }

    // Display the processed matrix on the root process (rank 0)
    if (rank == 0)
    {
        printf("\nMatrix After Processing\n\n");
        for (int i = 0; i < matrix_size; i++)
        {
            for (int j = 0; j < matrix_size; j++)
            {
                printf("%4d ", resultMatrix[i][j]); // Print the result matrix with consistent formatting
            }
            printf("\n");
        }
    }

    // Step 4: Clean up by releasing MPI groups and communicators
    MPI_Group_free(&even_group);   // Release memory allocated to the even group
    MPI_Group_free(&odd_group);    // Release memory allocated to the odd group
    MPI_Comm_free(&even_comm);     // Clean up communicator for the even group
    MPI_Comm_free(&odd_comm);      // Clean up communicator for the odd group
    MPI_Group_free(&global_group); // Free the base group used in this session

    MPI_Finalize(); // Finalize MPI operations and end the environment
    return 0;       // Exit the program successfully
}
