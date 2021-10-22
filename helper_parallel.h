
#ifndef HELPER_PARALLEL_H
#define HELPER_PARALLEL_H

#include <mpi.h>

template<typename T>
void mpi_print_array(T *array, size_t size, const char *msg="", int rank, int proc_num)
{
	for (int i=0; i<proc_num; i++) {
		if (i == rank) {
			fprintf(stdout, "%s ", msg);
			log_array_noendl(stdout, array, size);
			fprintf(stdout, "\n");
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}


#endif // HELPER_PARALLEL_H
