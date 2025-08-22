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

#ifndef TIMER_H
#define TIMER_H

#include "mpi.h"
#include <iostream>
#include <sstream>

namespace CabanaPD
{
class Timer
{
    double _time = 0.0;
    double _start_time = 0.0;
    double _last_time = 0.0;
    double _max_time = 0.0;
    double _min_time = 0.0;
    int _num_calls = 0;
    bool _running = false;

  public:
    void start()
    {
        if ( _running )
            throw std::runtime_error( "Timer already running" );

        _start_time = MPI_Wtime();
        _running = true;
    }
    void stop()
    {
        if ( !_running )
            throw std::runtime_error( "Timer not running." );

        _last_time = MPI_Wtime() - _start_time;
        _time += _last_time;
        _num_calls++;
        _running = false;
    }
    void reset() { _time = 0.0; }
    void set( const double val )
    {
        _time = val;
        _last_time = val;
    }
    bool running() { return _running; }
    auto time() { return _time; }
    auto minTime() { return _min_time; }
    auto maxTime() { return _max_time; }
    auto numCalls() { return _num_calls; }
    auto lastTime() { return _last_time; }

    void reduceMPI()
    {
        MPI_Allreduce( &_time, &_max_time, 1, MPI_DOUBLE, MPI_MAX,
                       MPI_COMM_WORLD );
        MPI_Allreduce( &_time, &_min_time, 1, MPI_DOUBLE, MPI_MIN,
                       MPI_COMM_WORLD );
    }
};
} // namespace CabanaPD

#endif
