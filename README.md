# Iterated-Prisoner-s-Dilemma-Framework
Framework for Iterated Prisoner's Dilemma with Memory where agents have memory, continuous gradient of cooperation probabilities and forgetting strategies.

- agent.py: Agent class that plays against each other. They have memory and forgetting strategy.
- IPD.py: Creates environment and agents. Runs a single iterated prisoner's dilemma with memory with given parameteres. Returns the results.
- main.py: Crates multiple environments each with its own parameters, accumulates the results and reports in form of csv. Uses parallelism for parallel execution of distinct realizations.