#ifndef COMM_H
#define COMM_H

#include <string>

namespace CabanaPD
{

class Comm
{
  public:
    int mpi_size;
    int mpi_rank;

    Comm()
    {
        MPI_Comm_size( MPI_COMM_WORLD, &mpi_size );
        MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
    }
    ~Comm() {}
};

} // namespace CabanaPD

#endif
