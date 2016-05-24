#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Program parameters */
#define MAX_SIZE 5000 // Max value of N
int N = 10;                // Matrix size

/* Matrices and vectors */
float A[MAX_SIZE * MAX_SIZE];
float B[MAX_SIZE];
float X[MAX_SIZE];
float **P; //partial A send to other processors
float **Q; //partial B send to other processors

/* Debug */
//#define DEBUG

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
	for (row = 0; row < N; row++) {
		for (col = 0; col < N; col++) {
			A[N * row + col] = (float)rand() / 32768.0;
		}
		B[row] = (float)rand() / 32768.0;
		X[row] = 0.0;
	}
}

/* Prepare data for each processor */
void prepare_data(int num_procs) {
	DEBUG_PRINT((stdout, "\nPreparing data...\n"));
	P = (float **)malloc(num_procs * sizeof(float *));
	Q = (float **)malloc(num_procs * sizeof(float *));
	int i;
	for (i = 0; i < num_procs; i++) {
		P[i] = (float *)malloc(((N + num_procs - 1) / num_procs) * N * sizeof(float));
		Q[i] = (float *)malloc(((N + num_procs - 1) / num_procs) * sizeof(float));
	}
	for (i = 0; i < N; i++) {
		memcpy(P[i % num_procs] + (N * (i / num_procs)), A + N * i, N * sizeof(float));
		*(Q[i % num_procs] + i / num_procs) = B[i];
	}
}

void scatter_data(int id, int num_procs) {
	DEBUG_PRINT((stdout, "\nP%d Scattering data...\n", id));
	// broadcast N to all processors
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	// scatter P[i] and B
	int h = (N + num_procs - 1) / num_procs;
	if (id == 0) {
		int i;
		memcpy((void *) A, (void *)P[0], h * N * sizeof(float));
		memcpy((void *) B, (void *)Q[0], N * sizeof(float));
		for (i = 1; i < num_procs; i++) {
			MPI_Send(P[i], N * h, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
			MPI_Send(Q[i], h, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
		}
	} else {
		MPI_Status status;
		MPI_Recv(A, N * h, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(B, h, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
	}
	DEBUG_PRINT((stdout, "P%d exited scatter...\n", id));
}

void gather_data(int id, int num_procs) {
	DEBUG_PRINT((stdout, "\nP%d gathering data...\n", id));
	int h = (N + num_procs - 1) / num_procs;
	if (id == 0) {
		// gather data from itself
		memcpy(P[0], A, N * h * sizeof(float));
		// gather data from other processors
		int i;
		MPI_Status status;
		for (i = 1; i < num_procs; i++) {
			MPI_Recv(P[i], N * h, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
			MPI_Recv(Q[i], h, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &status);
		}
	} else {
		MPI_Send(A, N * h, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
		MPI_Send(B, h, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
	}

	// re-assemble A
	if (id == 0) {
		int i;
		for (i = 0; i < N; i++) {
			memcpy(A + i * N, P[i % num_procs] + (N * (i / num_procs)), N * sizeof(float));
			B[i] = *(Q[i % num_procs] + (i / num_procs));
		}
	}
}


void gauss(int id, int num_procs) {
	int iter;
	float *ref_a;
	float *ref_b;
	ref_a = (float *)malloc(N * sizeof(float));
	ref_b = (float *)malloc(sizeof(float));
	for (iter = 0; iter < N; iter++) {
		// broadcast refence to all processors
		if (id == iter % num_procs) {
			memcpy(ref_a, A + N * (iter / num_procs), N * sizeof(float));
			MPI_Bcast(A + N * (iter / num_procs), N, MPI_FLOAT, iter % num_procs, MPI_COMM_WORLD);
			MPI_Bcast(B + iter / num_procs, 1, MPI_FLOAT, iter % num_procs, MPI_COMM_WORLD);
		} else {
			MPI_Bcast(ref_a, N, MPI_FLOAT, iter % num_procs, MPI_COMM_WORLD);
			MPI_Bcast(ref_b, 1, MPI_FLOAT, iter % num_procs, MPI_COMM_WORLD);
		}
		// gauss elimination
		int start_line = (iter + num_procs - id) / num_procs;
		DEBUG_PRINT((stdout, "P%d iter: %d start_line: %d\n", id, iter, start_line));
		int end_line = N / num_procs + ((id < (N % num_procs)) ? 1 : 0);
		DEBUG_PRINT((stdout, "P%d iter: %d end_line: %d\n", id, iter, end_line));
		int i, j;
		for (i = start_line; i < end_line; i++) {
			float c = ref_a[iter] / A[i * N + iter];
			for (j = iter; j < N; j++) {
				A[i * N + j] *= c;
				A[i * N + j] -= ref_a[j];
			}
			B[i] *= c;
			B[i] -= *ref_b;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	int i;
	for (i = 0; i < num_procs; i++) {
		if (i == id) {
			int j;
			DEBUG_PRINT((stdout, "P%d ref: ", id));
			for (j = 0; j < N; j++) {
				DEBUG_PRINT((stdout, "%f ", ref_a[j]));
			}
			DEBUG_PRINT((stdout, "\n"));
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	 free(ref_a);
	 free(ref_b);
}

/* Print the golabl matrix A and B */
void print_matrix() {
	int row, col;
	DEBUG_PRINT((stdout, "\nA =\n\t"));
	for (row = 0; row < N; row++) {
		for (col = 0; col < N; col++) {
			DEBUG_PRINT((stdout, "%5.2f%s", A[row * N + col],
					(col < N-1) ? ", " : ";\n\t"));
		}
	}
	DEBUG_PRINT((stdout, "\nB = ["));
	for (col = 0; col < N; col++) {
		DEBUG_PRINT((stdout, "%5.2f%s", B[col], (col < N-1) ? "; " : "]\n"));
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

void print_P(int id, int num_procs) {
	int i, j, k;
	if (id == 0) {
		for (i = 0; i < num_procs; i++) {
			DEBUG_PRINT((stdout, "\nP[%d]=\n\t", i));
			for (j = 0; j < (N + num_procs - 1) / num_procs; j++) {
				for (k = 0; k < N; k++) {
					DEBUG_PRINT((stdout, "%5.2f%s", P[i][j * N + k],
								(k < N - 1) ? ", " : ";\n\t"));
				}
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
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
		prepare_data(num_procs);
		// clear A & B
		memset(A, 0, MAX_SIZE * MAX_SIZE * sizeof(float));
		memset(B, 0, MAX_SIZE * sizeof(float));
	}

	double start_time = 0;
	if (id == 0) {
		start_time = MPI_Wtime();
	}

	scatter_data(id, num_procs);

	MPI_Barrier(MPI_COMM_WORLD);

	gauss(id, num_procs);

	//print_all_matrix(id, num_procs);

	gather_data(id, num_procs);

	double end_time = 0;
	if (id == 0) {
		end_time = MPI_Wtime();
		fprintf(stdout, "Running time: %f", end_time - start_time);
	}

	//print_all_matrix(id, num_procs);

	//print_P(id, num_procs);
		
	// back substitution
	if (id == 0) {
		int row, col;
		for (row = N - 1; row >= 0; row--) {
			X[row] = B[row];
			for (col = N - 1; col > row; col--) {
				X[row] -= A[row * N + col] * X[col];
			}
			X[row] /= A[row * N + row];
		}

		int i;
		DEBUG_PRINT((stdout, "X:"));
		for (i = 0; i < N; i++) {
			DEBUG_PRINT((stdout, "%5.2f ", X[i]));
		}
		DEBUG_PRINT((stdout, "\n"));
	}

	MPI_Finalize();

	return 0;
}
