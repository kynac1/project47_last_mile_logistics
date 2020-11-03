# Project #47: Dealing with uncertainty in last-mile logistics

## Setup

In this repository, we include a python library for running simulations, as well as some data and some experiments. See [installation](/Installation.md) for a guide to getting the project running.

Some functions require a routing server, for calculating distances and times between locations. This can be done through Google Maps, but the code to do this is relatively untested. The recommended approach is to run a local OSRM server. For a guide to downloading and running that dependency, see [Instructions for OSRM server](/Instructions%20for%20OSRM%20Server.md).

## Contents of the repository

- data : This contains some of the source data files we use for generating customer distributions.
- experiments : Final output files from running simulations, plus the scripts we used to generate that data.
- ortools_benchmarks : The code and output for running benchmarks of the ortools vehicle routing solver.
- project47 : A python library that contains all the reusable code for running simulations
  - customer.py : A customer object for testing if deliveries were successful. Also contains its position.
  - data.py : For generating latitudes and longitudes, and finding the road distances between them.
  - flp_data_vis.py : Helps for visualisation of solutions to the facility location problem.
  - flp_data.py : Helper functions for setting up the facility location problem.
  - flp_func.py : Solver for the facility location problem.
  - flp_vis.ipynb : Python notebook for facility location problem visualisation.
  - multiday_simulation.py : Contains code for running multiple days of the simulation, tying together all the other components.
  - routing.py : Helper functions for working with ortools to solve the vehicle routing problem
  - simulation.py : Functions for simulating a single day, and solving the hamiltonian shortest path back to the depo.
- tests : A unit testing suite. Not all tests are still useful, but should all pass.

# Initial project brief

## Description:

In the past 10-years e-commerce has expanded rapidly, leading to tremedous increases in goods being delivered directly to consumers' homes. This growth in demand is expected to accelerate over the next decade, so it is crucial that planning and logistics systems are developed to manage this increased demand.

In this project we will consider how different forms of uncertainty influences performance for delivery companies, and develop methods to handle this uncertainty.

## Outcome:

- Data analysis of previous order data and Census data to project future demand, by area.

- Implementation of Vehicle Routing Problem with time windows, modelling travel time and fuel usage.

- Approximate dynamic programming methods to find optimal delivery schedules.

- A simulation tool to evaluate performance of schedules.

## Prerequisites

- ENGSCI760

## Specialisations

- Engineering Science
- Categories
- Operations Research
- Transportation Modelling

## Supervisor

Tony Downward

## Co-supervisor

David Robb
