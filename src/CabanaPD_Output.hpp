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
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include <mpi.h>

#include <CabanaPD_Types.hpp>

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
        throw std::runtime_error( "Aborting after error." );
    }
}

template <class t_stream, class t_head, class... t_tail>
void log_err( t_stream& stream, t_head&& head, t_tail&&... tail )
{
    if ( print_rank() )
        stream << head;
    log_err( stream, std::forward<t_tail>( tail )... );
}

void checkParticleCount( std::size_t initial, std::size_t current,
                         std::string name )
{
    if ( initial != current )
        log_err( std::cout, "\nParticle size (", std::to_string( current ),
                 ") does not match size when ", name, " was created (",
                 std::to_string( initial ),
                 ").\n Likely a slice() call is missing." );
}

/******************************************************************************
 Timing output
******************************************************************************/

template <typename ForceType, typename ContactType, typename SFINAE = void>
struct TimingOutput;

template <typename ForceType>
struct TimingOutput<
    ForceType, NoContact,
    typename std::enable_if<( !is_contact<ForceType>::value )>::type>
{
    void header( const std::string output_file, const double init )
    {
        std::ofstream out( output_file, std::ofstream::app );
        log( out, "Running PD only.\n" );
        log( out, "Init-Time(s): ", init );
        if constexpr ( is_fracture<ForceType>::value )
            log( out,
                 "#Timestep/Total-steps Simulation-time Total-strain-energy "
                 "Total-Damage Cumulative-Time(s) Force-Time(s) Comm-Time(s) "
                 "Integrate-Time(s) Energy-Time(s) Output-Time(s) "
                 "Particle*steps/s" );
        else
            log( out,
                 "#Timestep/Total-steps Simulation-time Total-strain-energy "
                 "Cumulative-Time(s) Force-Time(s) Comm-Time(s) "
                 "Integrate-Time(s) Energy-Time(s) Output-Time(s) "
                 "Particle*steps/s" );
        out.close();
    }

    void step( const std::string output_file, const int step,
               const int num_steps, const double dt, const double W,
               [[maybe_unused]] const double relative_damage,
               const double total, const double force, const double,
               const double, const double comm, const double integrate,
               const double energy, const double output,
               const double p_steps_per_sec )
    {
        std::ofstream out( output_file, std::ofstream::app );
        log( std::cout, step, "/", num_steps, " ", std::scientific,
             std::setprecision( 2 ), step * dt );

        if constexpr ( is_fracture<ForceType>::value )
            log( out, std::fixed, std::setprecision( 6 ), step, "/", num_steps,
                 " ", std::scientific, std::setprecision( 2 ), step * dt, " ",
                 W, " ", relative_damage, " ", std::fixed, total, " ", force,
                 " ", comm, " ", integrate, " ", energy, " ", output, " ",
                 std::scientific, p_steps_per_sec );
        else
            log( out, std::fixed, std::setprecision( 6 ), step, "/", num_steps,
                 " ", std::scientific, std::setprecision( 2 ), step * dt, " ",
                 W, " ", std::fixed, total, " ", force, " ", comm, " ",
                 integrate, " ", energy, " ", output, " ", std::scientific,
                 p_steps_per_sec );
        out.close();
    }

    void final( const std::string output_file, const int size,
                const int global_particles, const double total,
                const double force, const double, const double neigh,
                const double, const double comm, const double integrate,
                const double energy, const double output, const double other,
                const double steps_per_sec, const double p_steps_per_sec )
    {
        std::ofstream out( output_file, std::ofstream::app );
        log( out, std::fixed, std::setprecision( 2 ),
             "\n#Procs Particles | Total Force Neighbor Comm Integrate Energy "
             "Output Other |\n",
             size, " ", global_particles, " | \t", total, " ", force, " ",
             neigh, " ", comm, " ", integrate, " ", energy, " ", output, " ",
             other, " | PERFORMANCE\n", std::fixed, size, " ", global_particles,
             " | \t", 1.0, " ", force / total, " ", neigh / total, " ",
             comm / total, " ", integrate / total, " ", energy / total, " ",
             output / total, " ", other / total, " | FRACTION\n\n",
             "#Steps/s Particle-steps/s Particle-steps/proc/s\n",
             std::scientific, steps_per_sec, " ", p_steps_per_sec, " ",
             p_steps_per_sec / size );
        out.close();
    }
};

template <typename ContactType>
struct TimingOutput<
    ContactType, NoContact,
    typename std::enable_if<( is_contact<ContactType>::value )>::type>
{
    void header( const std::string output_file, const double init )
    {
        std::ofstream out( output_file, std::ofstream::app );
        log( out, "Running DEM only.\n" );
        log( out, "Init-Time(s): ", init );
        log( out, "#Timestep/Total-steps Simulation-time Cumulative-Time(s) "
                  "Force-Time(s) Neighbor-Time(s) Comm-Time(s) "
                  "Integrate-Time(s) Energy-Time(s) Output-Time(s) "
                  "Particle*steps/s" );
        out.close();
    }

    void step( const std::string output_file, const int step,
               const int num_steps, const double dt, const double, const double,
               const double total, const double force, const double,
               const double neigh, const double comm, const double integrate,
               const double energy, const double output,
               const double p_steps_per_sec )
    {
        std::ofstream out( output_file, std::ofstream::app );
        log( std::cout, step, "/", num_steps, " ", std::scientific,
             std::setprecision( 2 ), step * dt );

        log( out, std::fixed, std::setprecision( 6 ), step, "/", num_steps, " ",
             std::scientific, std::setprecision( 2 ), step * dt, " ",
             std::fixed, total, " ", force, " ", neigh, " ", comm, " ",
             integrate, " ", energy, " ", output, " ", std::scientific,
             p_steps_per_sec );
        out.close();
    }

    void final( const std::string output_file, const int size,
                const int global_particles, const double total,
                const double force, const double, const double,
                const double neigh, const double comm, const double integrate,
                const double, const double output, const double other,
                const double steps_per_sec, const double p_steps_per_sec )
    {
        std::ofstream out( output_file, std::ofstream::app );
        log( out, std::fixed, std::setprecision( 2 ),
             "\n#Procs Particles | Total Force Neighbor Comm Integrate Output "
             "Other |\n",
             size, " ", global_particles, " | \t", total, " ", force, " ",
             neigh, " ", comm, " ", integrate, " ", output, " ", other,
             " | PERFORMANCE\n", std::fixed, size, " ", global_particles,
             " | \t", 1.0, " ", force / total, " ", neigh / total, " ",
             comm / total, " ", integrate / total, " ", output / total, " ",
             other / total, " | FRACTION\n\n",
             "#Steps/s Particle-steps/s Particle-steps/proc/s\n",
             std::scientific, steps_per_sec, " ", p_steps_per_sec, " ",
             p_steps_per_sec / size );
        out.close();
    }
};

template <typename ForceType, typename ContactType>
struct TimingOutput<
    ForceType, ContactType,
    typename std::enable_if<( is_contact<ContactType>::value )>::type>
{
    void header( const std::string output_file, const double init )
    {
        std::ofstream out( output_file, std::ofstream::app );
        log( out, "Running hybrid DEM-PD.\n" );
        log( out, "Init-Time(s): ", init );
        if constexpr ( is_fracture<ForceType>::value )
            log( out,
                 "#Timestep/Total-steps Simulation-time Total-strain-energy "
                 "Total-Damage Cumulative-Time(s) Force-Time(s) "
                 "Force-Contact-Time(s) Neighbor-Contact-Time(s) Comm-Time(s) "
                 "Integrate-Time(s) Energy-Time(s) Output-Time(s) "
                 "Particle*steps/s" );
        else
            log( out,
                 "#Timestep/Total-steps Simulation-time Total-strain-energy "
                 "Cumulative-Time(s) Force-Time(s) Force-Contact-Time(s) "
                 "Neighbor-Contact-Time(s) Comm-Time(s) "
                 "Integrate-Time(s) Energy-Time(s) Output-Time(s) "
                 "Particle*steps/s" );
        out.close();
    }

    void step( const std::string output_file, const int step,
               const int num_steps, const double dt, const double W,
               [[maybe_unused]] const double relative_damage,
               const double total, const double force, const double contact,
               const double neigh, const double comm, const double integrate,
               const double energy, const double output,
               const double p_steps_per_sec )
    {
        std::ofstream out( output_file, std::ofstream::app );
        log( std::cout, step, "/", num_steps, " ", std::scientific,
             std::setprecision( 2 ), step * dt );

        if constexpr ( is_fracture<ForceType>::value )
            log( out, std::fixed, std::setprecision( 6 ), step, "/", num_steps,
                 " ", std::scientific, std::setprecision( 2 ), step * dt, " ",
                 W, " ", relative_damage, " ", std::fixed, total, " ", force,
                 " ", contact, " ", neigh, " ", comm, " ", integrate, " ",
                 energy, " ", output, " ", std::scientific, p_steps_per_sec );
        else
            log( out, std::fixed, std::setprecision( 6 ), step, "/", num_steps,
                 " ", std::scientific, std::setprecision( 2 ), step * dt, " ",
                 W, " ", std::fixed, total, " ", force, " ", contact, " ",
                 neigh, " ", comm, " ", integrate, " ", energy, " ", output,
                 " ", std::scientific, p_steps_per_sec );
        out.close();
    }

    void final( const std::string output_file, const int size,
                const int global_particles, const double total,
                const double force, const double contact_force,
                const double neigh, const double contact_neigh,
                const double comm, const double integrate, const double energy,
                const double output, const double other,
                const double steps_per_sec, const double p_steps_per_sec )
    {
        std::ofstream out( output_file, std::ofstream::app );
        log( out, std::fixed, std::setprecision( 2 ),
             "\n#Procs Particles | Total Force Contact-Force Neighbor "
             "Contact-Neighbor Comm Integrate Energy Output Other |\n",
             size, " ", global_particles, " | \t", total, " ", force, " ",
             contact_force, " ", neigh, " ", contact_neigh, " ", comm, " ",
             integrate, " ", energy, " ", output, " ", other,
             " | PERFORMANCE\n", std::fixed, size, " ", global_particles,
             " | \t", 1.0, " ", force / total, " ", contact_force / total, " ",
             neigh / total, " ", contact_neigh / total, " ", comm / total, " ",
             integrate / total, " ", energy / total, " ", output / total, " ",
             other / total, " | FRACTION\n\n",
             "#Steps/s Particle-steps/s Particle-steps/proc/s\n",
             std::scientific, steps_per_sec, " ", p_steps_per_sec, " ",
             p_steps_per_sec / size );
        out.close();
    }
};
} // namespace CabanaPD

#endif
