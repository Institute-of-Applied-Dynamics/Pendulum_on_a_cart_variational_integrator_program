# Solving a pendulum on a cart optimal control problem


This code simulates the pendulum on a cart optimal control problem (OCP) with dynamical constraints given in the form of forced Euler-Lagrange equations.

The OCP is solved using a new Lagrangian approach defined in [1], using a low order family of numerical integrators [2].


The code allows to generate solutions for two specific choices of the OCP in the form of a pendulum inversion and a cart translation, using the jupyter notebook 'syimpy_implementation_paper_simu.ipynb'

The notebook 'Analysis_file.ipynb' contains the generation of the figures for paper [2] from the data created.

For the figures, 'syimpy_implementation_paper_simu.ipynb' needs to be used to create the following data:
 - Pendulum inversion
   - $\alpha=\gamma=0.5$
      - N = 150,300
   - $\alpha=\gamma = 1.0$
      - N= 30,70,150,250
 - Cart translation
   - $\alpha=\gamma=0.5$
      - N= 150, 300
   - $\alpha=\gamma=1.0$
      - N= 70,100,150,250

These parameters may be chosen by modifying 'N_choice', 'alpha_choice','beta_choice' and 'problem_choice' with the corresponding parameters in the modification section of the notebook.

For each parameter choice running the full notebook will generate and store the data.

Plotting of the figures can be accomplished then running the notebook 'Analysis_file.ipynb' after all the data has been generated and stored.










[1]: Konopik, M., Leyendecker, S., Maslovskaya, S., Ober-Blöbaum, S., and Sato Martin de Almagro, R. T.  A new Lagrangian approach to optimal control of second-order systems. 2025. arXiv: 2503.04466 [math.OC].

[2] Konopik, M., Leyendecker, S., Maslovskaya, S., Ober-Blöbaum, S., and Sato Martin de Almagro, R. T. On the variational discretization of optimal control problems for Lagrangian dynamics. 2025. ResearchSquare:  https://doi.org/10.21203/rs.3.rs-6566751/v1

