# Cube with linear temperature profile

This regression test consists of the simplest possible test of macroscopic, uniaxial temperature transport.
To remove the mechanics from this equation, the coefficient of thermal expansion $\alpha = 0$.

## Purpose

This regression test validates that the macroscopic temperature transport behavior modeled with the peridynamic models is consistent with analytic theory.
It uses an analytical reference temperature profile that is linear in x-direction.

## Case setup

The regression test is kept as simple as possible. It uses a 1-D, linear, temperature profile in the x-direction.

### Domain

It uses a simple cubic domain with one density and material type.
The particles are spaced equidistantly in each direction of the 3D space.

### Boundary Conditions

To eliminate surface effects, the cubic domain is created with an additional $\delta$-thick layer of particles in each direction.
We will call this layer the boundary layer in the following text.

In all of this boundary layer we apply a Dirichlet boundary condition of the temperature $T$ on the $\delta$ layer. Here the following temperature is imposed with $\Delta T = 1K$:
$$
T = T_{ref} + \Delta T * x_0
$$

The rest of the domain is initialized with $T=T_{ref}$.
This boundary represents the analytical solution to a uni-axial linear temperature profile in x-direction.
Boundary conditions are applied over the first half of the simulation time, the rest is used to ensure a stationary solution.

### Solver

As only the thermal computation is relevant here, it uses the default forward euler timestepping for the temperature equation.

## Evaluation

In general, the regression test is only evaluating the final solution and only in a center regrion of the cube. This eliminates surface and boundary effects.
The test compares the temperature with the analytical reference.

### Reference solution

Over time the solution for $T$ inside the domain will approach the linear profile that is imposed on the boundary.
Therefore, we check if $T \approx T_{ref} + \Delta T * x_0$ for all particles in the boundary region. The admissible tolerance is $10^{-16}K$
