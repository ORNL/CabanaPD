/****************************************************************************
 * Copyright (c) 2022 by Oak Ridge National Laboratory                      *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of CabanaPD. CabanaPD is distributed under a           *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef OUTPUT_H
#define OUTPUT_H

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>

#include <mpi.h>

namespace CabanaPD
{

inline bool print_rank()
{
    int proc_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &proc_rank );
    return proc_rank == 0;
}

template <class t_stream, class t_last>
void log( t_stream& stream, t_last&& last )
{
    if ( print_rank() )
        stream << last << std::endl;
}

template <class t_stream, class t_head, class... t_tail>
void log( t_stream& stream, t_head&& head, t_tail&&... tail )
{
    if ( print_rank() )
        stream << head;
    log( stream, std::forward<t_tail>( tail )... );
}

template <class t_stream, class t_last>
void log_err( t_stream& stream, t_last&& last )
{
    if ( print_rank() )
    {
        stream << last << std::endl;
        throw std::runtime_error(
            "Aborting after error from input. See error file." );
    }
}

template <class t_stream, class t_head, class... t_tail>
void log_err( t_stream& stream, t_head&& head, t_tail&&... tail )
{
    if ( print_rank() )
        stream << head;
    log_err( stream, std::forward<t_tail>( tail )... );
}

} // namespace CabanaPD

#endif
