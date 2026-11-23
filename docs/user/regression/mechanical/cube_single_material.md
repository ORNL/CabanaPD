# Cube with single material

This regression test consists of the simplest possible test of macroscopic material behavior, material in uniaxial, constant strain.

## Purpose

This regression test validates that the macroscopic material behavior modeled with the peridynamic models is consistent with analytic theory.
Furthermore, it establishes the baseline for accuracy for the displacement and stresses with respect to the used material parameters.
Therefore, it uses an analytical reference solution to compare the numerical results to.

## Case setup

The regression test is kept as simple as possible. It uses a 1-D, constant, axial strain $\epsilon_{x}$ in the x-direction and $\nu = 0.25$.

### Domain

It uses a simple cubic domain with one density and material type.
The particles are spaced equidistantly in each direction of the 3D space.

### Boundary Conditions

To eliminate surface effects, the cubic domain is created with an additional $\delta$-thick layer of particles in each direction.
We will call this layer the boundary layer in the following text.
In all of this boundary layer we apply a Dirichlet boundary condition of the displacement on the $\delta$ layer. Here the following displacement is imposed:
$$
u_0 = \epsilon_{x} * x_0
$$
$$
u_1 = -\nu \epsilon_{x} * x_1
$$
$$
u_2 = -\nu \epsilon_{x} * x_2
$$

This boundary represents the analytical solution to uni-axial strain in x-direction.
Boundary conditions are applied over the first half of the simulation time, the rest is used to ensure a stationary solution.

### Solver

As the stresses of the solution will be evaluated, this regression test is using the adaptive dynamic relaxation to compute a stationary solutioin. It is observed that the solver was not able to reduce the force residual to values $<10^{-11}$.
This resulted in the stresses having a residual error of $<10^{-1}$.

## Evaluation

In general, the regression test is only evaluating the final solution and only in a center regrion of the cube. This eliminates surface and boundary effects.
The test compares the displacement in all 3 directions and the resulting stresses to the reference solution.

### Reference solution

We use the uni-axial strain $\epsilon_{x}$ to directly validate the displacements as stated in the section on boundary conditions.
The displacements in the center region of interest of the cube must be the same as the analytical solution used in the boundary layer.

To validate the stresses, we use Hooke's law in 3D:
$$
\sigma_{ij} = 2G \epsilon_{ij} + \frac{E \nu}{(1+\nu)(1-2\nu)} \sum_{k=1}^{3} \epsilon_{kk} * I_3
$$

with the strain tensor that is imposed on the cube:
$\epsilon_{ij} = \begin{bmatrix} \epsilon_x & 0 & 0 \\ 0 & -\nu \epsilon_x & 0 \\ 0 & 0 & -\nu \epsilon_x \end{bmatrix}$

this results in:
$\sigma_{ij} = \begin{bmatrix} E \epsilon_x & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}$

this calculation used $G = \frac{E}{2(1+nu)}$ for isotropic materials.
