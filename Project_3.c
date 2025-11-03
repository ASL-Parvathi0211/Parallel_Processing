#include <stdio.h>

#include <stdlib.h>

#include <mpi.h>

#define generate_data(i, j)(i) + (j) * (j) // Formula for generating matrix elements

int main(int argc, char ** argv) {
  int i, j, pid, np, mtag;
  int data[100][100], row_sum[50]; // Array for storing the matrix and the computed row sums
  int split_data[30][100]; // Temporary storage for the second chunk of 30 rows
  MPI_Status status; // Struct to store MPI status information
  MPI_Request req_s2, req_r, req_split; // Handles for non-blocking communication

  MPI_Init( & argc, & argv); // Initialize the MPI environment
  MPI_Comm_rank(MPI_COMM_WORLD, & pid); // Get the process ID
  MPI_Comm_size(MPI_COMM_WORLD, & np); // Get the total number of processes

  if (pid == 0) { // Code executed by process 0
    // Generate the first 20 rows of the matrix
    for (i = 0; i < 20; i++) {
      for (j = 0; j < 100; j++) {
        data[i][j] = generate_data(i, j); // Populate the matrix with data for rows 0 to 19
      }
    }

    // Generate the next 30 rows of the matrix
    for (i = 20; i < 50; i++) {
      for (j = 0; j < 100; j++) {
        data[i][j] = generate_data(i, j); // Populate the matrix with data for rows 20 to 49
        split_data[i - 20][j] = data[i][j]; // Store these rows temporarily in split_data
      }
    }

    // Send the first 20 rows to process 1 using non-blocking send
    MPI_Isend(data, 2000, MPI_INT, 1, 20, MPI_COMM_WORLD, & req_split);

    // Send the next 30 rows to process 1 using non-blocking send
    MPI_Isend(split_data, 3000, MPI_INT, 1, 30, MPI_COMM_WORLD, & req_s2);

    // Wait until the second transmission is complete
    MPI_Wait( & req_s2, & status);

    // Receive the row sums from process 1
    mtag = 2; // Message tag to distinguish this receive
    MPI_Recv(row_sum, 50, MPI_INT, 1, mtag, MPI_COMM_WORLD, & status);

    // Output the received row sums
    for (i = 0; i < 50; i++) {
      printf(" %d ", row_sum[i]);
      if (i % 10 == 9) printf("\n");
    }
  } else { // Code executed by process 1
    // Receive the first 20 rows from process 0
    MPI_Recv(data, 2000, MPI_INT, 0, 20, MPI_COMM_WORLD, & status);

    // Start receiving the next 30 rows from process 0 using non-blocking receive
    MPI_Irecv(split_data, 3000, MPI_INT, 0, 30, MPI_COMM_WORLD, & req_r);

    // Compute the row sums for the first 20 rows
    for (i = 0; i < 20; i++) {
      row_sum[i] = 0;
      for (j = 0; j < 100; j++) {
        row_sum[i] += data[i][j]; // Calculate the sum of the elements in each row
      }
    }

    // Wait until the remaining 30 rows have been received
    MPI_Wait( & req_r, & status);

    // Compute the row sums for the next 30 rows
    for (i = 0; i < 30; i++) {
      row_sum[20 + i] = 0;
      for (j = 0; j < 100; j++) {
        row_sum[20 + i] += split_data[i][j]; // Compute the sum for rows 20 to 49
      }
    }

    // Send the computed row sums back to process 0
    mtag = 2;
    MPI_Send(row_sum, 50, MPI_INT, 0, mtag, MPI_COMM_WORLD);
  }

  MPI_Finalize(); // Terminate the MPI environment

  return 1;
}
