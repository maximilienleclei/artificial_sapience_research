# Introduction
## Premise
We see intelligence as the raw ability to solve problems. In contrast, we see wisdom as the ability to create and leverage high-order information in order to calibrate problem solving.

This repository constitutes our attempt to imbue AI systems with more of what we consider wisdom.

At the present stage (2026/05), this repository is geared towards drawing previously unextracted human behaviour characteristics out of human behaviour datasets, through an exotic interplay of computational methods.
## Strategy
### Motivation
Our belief is that, in practice, several conditions need to all be met in order for this extraction to start showing visible signs of success: imitation of human behaviour in its complete form, proper calibration of random search conditioned on gradient-based representations, etc (extensive details in later sections).

In order to meet all of these conditions, we believe that a multitude of methods need to be incorporated into a working solution. Several of these methods have, in isolation and in older forms, been scientifically peer-reviewed. However, many of them have not.

In order to account for time constraints, we will be taking the (somewhat) measured “leap of faith” of betting on years of conceptual design and experimentation directed towards this very purpose.
### Outline
We provide, starting from the second section of this document 1) a specification sheet that contains an appropriately detailed description of all necessary components for the implementation of a) the envisioned approach & b) baselines to compare against; and 2) a proposed rough path for experimentation.

These components are meant to provide a solid framework for a sufficiently capable AI agent to both 1) generate the full desired apparatus and 2) run extensive incremental experimentation in order to build any of the missing understanding that very well could be required in order to successfully combine all of these methods together.
## Central hypotheses
### Modern AI systems & their relation with high-order information
While modern AI systems exhibit increasingly more advanced and efficient problem-solving in a wide range of domains, they appear, to us, limited in their ability to create and understand high-order information such as matters of the human condition, societal dynamics, etc.

While the argument can be made that these systems are vastly different from us, it is also undeniable that they have also been exposed to an amount of information pertaining to these high-order concepts at a scale that no human has ever been, by far.

It thus appears to us that such wisdom is not going to naturally emerge from the current paradigm.
### The need for higher fidelity human behaviour imitation
In the pursuit of designing systems that exhibit wisdom, it appears that one has to make a judgement call with respect to a gradient that spans from pure imitation of existing behaviour all the way to independent discovery.

In our perspective, wisdom as it appears in humanity is an extremely advanced product of evolution. As a result, we consider it naive to practically hope for success from anything that distances itself from pure imitation of existing behaviour.

We thus believe that our best bet at embedding wisdom into computational systems is through the betterment of human behaviour imitation modeling*.

We propose to take a shot at that problem from different angles: collaborative adversarial imitation, unfiltered behaviour over time. etc.

* In our opinion, many organic life forms and their societies do show signs of wisdom, but their behaviour is practically quite a bit harder to turn digital.
### Random search as a missing component
#### Current paradigm
We observe that, over the past ~15 years, AI research has largely focused on exploring a wide range of priors within a gradient-based optimization dominated framework.

Gradient-based optimization methods are tightly coupled to the data distribution, funneling information from the data directly into the model’s representation space.
#### Proposed extension
Our hypothesis is that, given our non-omniscience, meaningful value remains unextracted when search and optimization are constrained to prior and data space.

We argue for the need to augment the optimization process through the exploration of random space.

The most popular random space exploration method is perhaps random search. It is an optimization method that is employed when we 1) do not have the necessary priors to successfully solve a task and 2) do not have a dataset to draw information from.

When we do have a dataset to draw information from however, it becomes natural to turn to evolutionary search. In evolutionary search, the representation space is also perturbed through random search, but data now plays a regularizing role in constraining this exploration.
#### Main constraints & remediations
From our humble perspective, random and evolutionary search permeate several conditions that, in practice, make their successful calibration quite exotic relative to modern gradient-based methods.

We introduce what we consider to be the two of these conditions, and how we think best to approach them.
##### The indirect influence of data
Given that, relative to gradient-based methods, data is retrograded from being a direct to an indirect optimization signal, valuable data information now gets assimilated into representation space at a much slower, less efficient and noisier pace.

We propose to mitigate this phenomenon by building on top of the modern paradigm. In practice, this means extracting information using gradient-based methods and operating evolutionary search in the realm of this extracted information.
##### Navigating random space
Perturbing representation space using random space is a delicate endeavour. With respect to our objectives, much of random space is undesired noise.

We thus need to calibrate our search to satisfy a delicate balance between representation construction and noise application. In the pursuit of that interest, we propose neural networks that we design for that balance.
# Specification sheet (in progress)
## Data
We make use of the following data setups, that are incrementally better fits to the goal of imitating human behaviour:

Environments:

Control tasks with <10 input values (e.g. position, velocity), <10 output values (e.g. actuator pressure)
Retro video games with ~2000 input values (RAM indices), ~10 output values (e.g. up, down, jump)
Retro video games with millions of input values (image stream), ~10 output values (e.g. up, down, jump)

For each of these environments we have human subject behaviour data. In all settings, this data is not stationary since the subjects’ behaviour evolves with experience. For the retro video games, we also have fMRI neuroimaging data of the subjects that corresponds to the collected behaviour data.


In each environment, we aim to create models that are able to accurately model the complete behaviour of our human subjects, meaning not only their behaviour at a particular skill level but also the way their behaviour evolves with experience. However, for the sake of troubleshooting, we ought to experiment our way up increasingly fitting behaviour, namely:


Fixed naive behaviour policy
Repeats a single action
Alternates between actions following a given pattern
Pre-trained capable RL agent policy
Subject “stable” behaviour (meaning we would need to find a particular time slice where behaviour statistics are “stable”)
Full subject behaviour
## Methods
### Standard deep learning
dw
### Neuroevolution
#### Core design
##### Introduction
The proposed evolutionary algorithm is an island-based genetic algorithm.

It maintains a single population of a fixed number `pop_size` of agents.
There are `num_islands` islands, each composed of the same `island_size` (`pop_size / num_islands`) number of agents.

Each agent has three distinct roles: generator, discriminator and mutator (more details below).

The algorithm loops every iteration through three stages: perturbation, evaluation and selection.
The generation of agents at iteration `<iter>` is `gen <iter>`.

During the perturbation stage, all agents are randomly mutated (no crossovers).
During the evaluation stage, all agents are quantitatively graded with a fitness on the given task.
##### Selection
Selection for both the islands and the population is achieved using 50% truncation selection.

This corresponds to ranking agents by fitness score, discarding the bottom 50% of agents and duplicating the top 50% of agents (to put it in illustrated language terms: selected agents become parents, produce two offspring, and erase themselves).

Per-island selection occurs every iteration. Every `island_merge_freq` iterations, the global population is ranked and 50% truncation selection occurs. New islands are then formed using the selected population at random.
#### Generational inheritance
Generational inheritance is a key component to make evolutionary search more efficient with respect to evaluation.

Its main purpose is to enable structure such that we can evolve over a desired evaluation time frame, no more no less.

In that framework, agents that are selected transfer to their two offspring not only their genotype but also their final environment state, final memory state, and cumulative lineage fitness scores.

During its own evaluation stage, the offspring thus slots in its parent’s memory state and resumes from its parent’s environment state. Its fitness score becomes the sum of the lineage fitness score and its own. The selection stage then accounts for that sum rather than the individual’s fitness.

The child is evaluated for `num_states` states, accumulating local reward. If termination occurs before E is exhausted under fixed E=k, reset the environment and use the remaining actions; otherwise descendants of terminal agents start from a fresh environment. After evaluation, store the child’s final env state and memory state, and set fitness = inherited fitness + local reward. Selection is based on this cumulative fitness.
#### Adversarial imitation
We now explain the generator and discriminator roles introduced earlier.




Every iteration, 

##### Application to generational inheritance

Generational inheritance is straight-forward to implement for the generator role.

However it is not so straightforward for the discriminator role, given that an agent’s chile will most likely not encounter the same generator 
#### Neural networks evolution
Each agent maintains one dynamic neural network. Its topology evolves alongside a small set of developmental parameters: hidden-node count, wiring pattern, local connection bias, expected grow/prune mutation counts, and the number of recurrent-style computation passes used for each input.

Each network is represented as a directed graph of nodes. The population is flattened into batched tensors for evaluation, while mutation remains graph-based and sequential.

##### Graph representation
The graph has three node roles:

- Input nodes are non-parametric nodes that receive external observations. There is one input node per observation dimension.
- Hidden nodes are mutable computational nodes. They can be added or removed by evolution. Each hidden node also has a fixed polarity, sampled uniformly at birth from `{-1, +1}`.
- Output nodes are mutable computational nodes whose standardized values are returned as the network output. There is one output node per action/output dimension.

Input and output nodes are created during initialization and are permanent. Hidden nodes are grown and pruned over evolutionary time. Every node has a mutable identifier, used for tensor indexing, and an immutable identifier, used to clone and reconstruct the graph after pruning has changed mutable positions.

Connections are incoming-node and outgoing-node references. Hidden and output nodes may have at most `MAX_INCOMING_CONNECTIONS = 3` incoming connections. Connections have no weights, signs, or other parameters: they only specify which node outputs are summed by the receiving node.

The graph maintains synchronized views of nodes by role and connection status:

- `all`: every node in mutable-index order.
- `input`, `hidden`, `output`: nodes grouped by role.
- `receiving`: nodes with incoming connections, repeated once per incoming connection.
- `emitting`: nodes with outgoing connections, repeated once per outgoing connection.
- `being_pruned`: a temporary recursion guard during cascading pruning.

##### Computation and normalization
Input nodes receive observation values. Hidden and output nodes compute an unweighted sum of up to three incoming standardized node values:

```text
x = sum(input_z_i)
z = standardize_per_node(x)
```

The network does not use connection weights, node biases, or activation functions. Instead, every node has persistent running standardization statistics:

- `n`: number of observed raw values.
- `mean`: running mean.
- `m2`: running sum of squared deviations.
- `x`: previous raw value.
- `z`: previous standardized output.

Standardization uses Welford's online algorithm. New non-zero values update the running statistics and return a z-score. Zeros and previous z-scores pass through without updating. A node returns zero until it has at least two samples, giving new structures a short warmup period before they emit meaningful standardized signals.

Hidden-node polarity is applied after standardization:

```text
hidden_output = polarity * z
```

Output nodes do not have polarity. They return their standardized value directly, keeping each output dimension stable while inhibitory or contrastive structure emerges inside the hidden graph.

These normalization statistics are learned network state, not episode-local hidden state. Resetting a network for a new episode therefore does not clear them.

##### Growth
Initialization creates all input and output nodes. After initialization, growth creates hidden nodes.

When a hidden node is grown, three connections are created:

- `in_node_1 -> new_node`
- `in_node_2 -> new_node`
- `new_node -> out_node_1`

Sampling first connects unused parts of the graph: the first incoming connection prioritizes input nodes that do not yet emit, and the outgoing connection prioritizes output nodes that do not yet receive. This helps early networks connect all inputs and outputs before growing denser internal structure.

When there are multiple candidates, sampling is graph-local: the sampler expands outward from a reference node and accepts candidates at each distance with probability `local_connectivity_probability`. Higher values encourage local modular wiring; lower values allow more global wiring.

If several grow mutations happen in one mutation call, they are chained: each new hidden node becomes the starting input node for the next growth operation.

##### Pruning
Pruning removes hidden nodes. Input and output nodes are never pruned.

When a hidden node is pruned, its incoming and outgoing connections, standardization state, and fixed polarity are removed. Later mutable identifiers are decremented so tensor indices stay contiguous. Pruning can cascade: if another hidden node loses all incoming or outgoing connections, it is pruned as well. A temporary `being_pruned` list prevents recursive pruning loops.

##### Mutation
Mutation first perturbs evolvable scalar parameters:

- `avg_num_grow_mutations` is multiplied by `1 + 0.01 * randn`.
- `avg_num_prune_mutations` is multiplied by `1 + 0.01 * randn`.
- `local_connectivity_probability` receives additive `0.01 * randn` noise and is clamped to `[0, 1]`.
- `num_network_passes_per_input` decreases by one with probability 1/100 when above one, and increases by one with probability 1/100.

Then the architecture is perturbed. Average grow/prune mutation values are converted into integer counts stochastically: the integer part is always used, and the fractional part is the probability of adding one more mutation. Pruning is applied first, growth second.

After mutation, the network regenerates computation tensors:

- `in_nodes_indices`: for each output/hidden node, the mutable identifiers of its incoming nodes, padded with `-1`.
- `hidden_polarities`: the fixed `-1` or `+1` polarity for each hidden node.

Mutable nodes are ordered as output nodes first, then hidden nodes. The incoming-node index tensor and polarity tensor both use this order.

##### Batched population computation
Although networks have heterogeneous topologies, the population is evaluated through one flattened tensor layout. The population-level forward call accepts `[num_nets, num_inputs]` and returns `[num_nets, num_outputs]`.

Before computation, the population is converted into batched lookup tensors:

- input-node start indices for each network in the flattened node tensor;
- flattened positions of all input nodes;
- flattened positions of all output nodes;
- flattened positions of all non-input nodes;
- a flattened incoming-node lookup table for every mutable node;
- all hidden-node polarities;
- a pass mask describing which networks should still update at each recurrent-style pass.

Index `0` in the flattened tensors is a dummy node that always outputs zero. Empty connection slots are clamped to this index so padded inputs can participate safely in vectorized gathers.

The forward pass proceeds as follows:

```text
1. Start from the previous standardized node outputs.
2. Insert the current observations into the flattened input-node positions.
3. Standardize the input nodes.
4. For each network pass:
   a. gather the three inputs for every mutable node;
   b. compute unweighted sums;
   c. standardize the updated node values;
   d. apply fixed polarity to hidden nodes;
   e. update only the mutable nodes whose network is still active for this pass.
5. Gather and reshape output nodes into [num_nets, num_outputs].
```

The number of passes is the maximum `num_network_passes_per_input` across the population. Networks with fewer passes are masked out after their own pass count, giving networks evolvable recurrent-like depth while preserving one batched computation path.

##### Resampling and cloning
Selection resamples the population by replacing each network with a clone of a selected parent. Cloning uses explicit graph serialization instead of generic deep copying because incoming/outgoing node references are circular and large evolved graphs can exceed Python's recursion limit.

Serialization stores each node's role, mutable identifier, immutable identifier, hidden-node polarity where applicable, incoming immutable identifiers, running standardization state, self-adapting mutation parameters, total nodes grown, and device placement. Reconstruction first creates all nodes, then reconnects incoming edges by immutable identifier.

Forward passes update the population's batched running-standardization tensor, so individual graph objects are stale during evaluation. Before mutation after a forward phase, the batched statistics are synchronized back into each network so offspring inherit the current normalization state.

##### Performance notes
Mutation is performed sequentially across networks, then tensor data is cached for fast batched preparation. Multiprocessing was rejected because individual mutations were too cheap relative to inter-process communication and serialization overhead. The current approach keeps graph mutation simple and uses tensor operations for forward computation.

### Generational inheritance




In the context of evolutionary adversarial generation, generator memory is straightforward to 

X Evolved network on top of gradient-based network

Modern AI policy network
Then have a network evolve akin to a parasite.
Starts by mapping from the output space back into the output space
Can start to go fetch from earlier layers after a certain point


X Objective: show that evolutionary algorithms can extract untapped value on top of gradient-based methods

	X Attempt showing in the context of static policy behaviour imitation, however it is likely
	that there is too little value left to extract.

	We have a dataset of human 

## Experimentation path
