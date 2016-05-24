#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

/* Program parameters */
#define MAX_SIZE 2000 // Max value of N
int N = 10;                // Matrix size

/* Matrices and vectors */
float A[MAX_SIZE][MAX_SIZE];
float B[MAX_SIZE];
float X[MAX_SIZE];

/* Debug */
#define DEBUG
#define PRINT_INPUT
#define PRINT_RESULT

#ifdef DEBUG
# define DEBUG_PRINT(msg) fprintf msg
#else
# define DEBUG_PRINT(msg) do {} while (0)
#endif

/* Check the input parameters */
void parameters(int argc, char **argv) {
	if (argc == 3) {
		srand(atoi(argv[2]));	
		N = atoi(argv[1]);
		if (N < 1 || N > MAX_SIZE) {
			fprintf(stdout, "N = %d is out of range.\n", N);
			exit(0);
		}
		DEBUG_PRINT((stdout, "Random seed = %d\n", atoi(argv[2])));
		DEBUG_PRINT((stdout, "Matrix demension N = %d\n", N));
	} else {
		fprintf(stdout, "Usage: %s <matirx_dimension> <random seed>\n", argv[0]);
		exit(0);
	}
}

/* Initialize global matrix A, B and X */
void initialize_inputs() {
	int row, col;

	DEBUG_PRINT((stdout, "\nInitializing...\n"));
	for (col = 0; col < N; col++) {
		for (row = 0; row < N; row++) {
			A[row][col] = (float)rand() / 32768.0;
		}
		B[col] = (float)rand() / 32768.0;
		X[col] = 0.0;
	}
}

/* Print the golabl matrix A and B */
void print_matrix() {
	int row, col;
	fprintf(stdout, "\nA =\n\t");
	for (row = 0; row < N; row++) {
		for (col = 0; col < N; col++) {
			fprintf(stdout, "%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
		}
	}
	printf("\nB = [");
	for (col = 0; col < N; col++) {
		fprintf(stdout, "%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
	}
}

/* Print the global matrix A and B in each processor in serial */
void print_all_matrix(int id, int num_procs) {
	int i;
	for (i = 0; i < num_procs; i++) {
		if (id == i) {
			DEBUG_PRINT((stdout, "This is process %d.\n", id));
			print_matrix();
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

int main(int argc, char **argv) {

	MPI_Init(&argc, &argv);
	int num_procs;
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	int id;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	int proc_name_len;
	char proc_name[MPI_MAX_PROCESSOR_NAME];
	MPI_Get_processor_name(proc_name, &proc_name_len);

	if (id == 0) {
		parameters(argc, argv);
		initialize_inputs();
	}

#ifdef PRINT_INPUT
	print_all_matrix(id, num_procs);
#endif

	// gauss();

	MPI_Finalize();

	return 0;
}
