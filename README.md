# Triolapse 👾

Triolapse is an exciting project that tackles the [No-Three-In-Line Problem](https://en.wikipedia.org/wiki/No-three-in-line_problem). It combines game theory, AI, and neural networks to create an intelligent player that thrives on strategy and decision-making. 🧠🎮

## The Challenge 🎯

Imagine a grid-based game where players take turns to place pieces. The goal is to fill the grid with as many pieces as possible, but here's the twist: you don't want to form three pieces in a line.

## How It Works 🚀

Triolapse employs the Monte Carlo Tree Search (MCTS) algorithm, a powerful tool in the AI arsenal, to make intelligent decisions. This algorithm simulates thousands of possible game states, selecting the best moves based on these simulations. The game's intelligence is further boosted by a neural network that evaluates game states and guides decision-making.

## Run It Yourself 🏃‍♂️

1. **Clone the Repository**: Start by cloning the Triolapse repository.
```bash
git clone https://github.com/vjdad4m/triolapse
```
2. **Run the Program**: Navigate to the project directory and run the main script.
```bash
cd triolapse
python3 triolapse/triolapse.py
```