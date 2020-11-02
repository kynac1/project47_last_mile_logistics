# Project #47: Dealing with uncertainty in last-mile logistics

## Guide to this repository

In this repository, we include a python library for running simulations, as well as some data and some experiments. See [installation](/Installation.md) for a guide to getting the project running.

Some functions require a routing server, for calculating distances and times between locations. This can be done through Google Maps, but the code to do this is relatively untested. The recommended approach is to run a local OSRM server. For a guide to downloading and running that dependency, see [Instructions for OSRM server](/Instructions%20for%20OSRM%20Server.md).


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
