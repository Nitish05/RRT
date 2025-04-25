# Rapidly-Exploring Random Tree (RRT) Variants Evaluation

## Overview
This repository provides implementations and comparisons of several Rapidly-Exploring Random Tree (RRT) variants for 2D path planning. The goal is to evaluate baseline RRT methods and introduce targeted optimizations to the Informed Quick RRT to achieve faster convergence and reduced node exploration.

## RRT Variants Evaluated
- **RRT**: Basic random tree expansion without optimality guarantees.
- **RRT***: Extends RRT with a rewiring procedure to approach optimal paths.
- **Informed RRT**: Uses heuristics to focus sampling within an ellipsoidal subset of the search space.
- **Informed RRT***: Combines the rewiring of RRT* with informed sampling.
- **Quick RRT**: Streamlines node insertion and nearest-neighbor queries for faster performance.
- **Informed Quick RRT**: Integrates informed sampling into the Quick RRT framework.
- **Modified Informed Quick RRT**: Adds heuristic-guided sampling density and optimized tree data structures for further speed-ups.

## Modifications to Informed Quick RRT
To enhance the Informed Quick RRT, the following changes were made:
1. **Adaptive Sampling Region**: Dynamically adjusts the informed ellipsoid based on the current best solution cost.
2. **KD-Tree Indexing**: Replaces brute-force nearest-neighbor searches with a balanced KD-Tree for O(log n) queries.
3. **Batch Node Insertion**: Groups multiple samples per iteration to reduce overhead in tree updates.

## Evaluation Metrics
Performance comparisons include:
- **Planning Time**: Wall-clock time to find an initial solution.
- **Node Count**: Number of nodes explored before solution discovery.
- **Path Cost**: Total length of the final planned path.
- **Convergence Rate**: Improvement of path cost over time.
