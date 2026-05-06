This is a single-page, carefully sectioned markdown document that details our research at various levels of abstraction.

It is optimized for both human and AI agent readability.

The tipping point motivation for creating this document was to fully decouple research methodology and experimentation away from a consistent codebase, which has only recently become legitimate.

This document has two main sections: `Overview` and `Specifications`.

We describe, in the `Overview` section, our high-level understanding and research reasoning so as to contextualize our `Specifications` section.

# Overview

We begin this section by sharing, in `Why artificial sapience?` our current understanding of both sapience and its relation with artificial intelligence.

We then proceed, in `Central hypotheses`, to introducing and motivating several hypotheses that shape the nature of our research direction.

Finally, in `Research strategy`, we share insights into how we operationalize our research endeavour.

## Why artificial sapience?

We motivate, in this sub-section, the value of artificial sapience in our artificial intelligence focused world.


### Sapience

Intelligence, wisdom and understanding are three advanced concepts that concern themselves with problem solving.

Just like most high-order concepts, all these concepts have neither universally agreed definitions nor clear boundaries with each other.

To the best of our knowledge, sapience is the term that best englobes the three terms in how they manifest in human beings.

### Intelligence

Our perspective is that intelligence is proportional to the likelihood of solving a given problem.

There is per-problem intelligence: an entity can be very good at solving very specific problems but very bad at solving everything else. In that case, it is narrowly very intelligent.

More broadly valuable however is general intelligence. Human beings for instance are able to solve a wide range of problems.

### Sacrificing sapience for the sake of greater intelligence

In the pursuit of increasing the likelihood of solving a given set of problems, greater intelligence is willing to forgo certain endeavours, such as 1) considering information pertaining to the greater context that the set of problems is in (wisdom) & 2) building comprehension that is not strictly required to “solve” the tasks (understanding).


### The artificial intelligence imbalance

We believe that there is an imbalance in the world of artificial systems wherein intelligence is prioritized at the expense of the other concepts that make up sapience.

This repository represents our attempt at a contribution in the pursuit of restoring this balance.

## Central hypotheses

We share, in this sub-section, the three core hypotheses that motivate our research endeavours.


### Modern AI systems, wisdom & understanding

While modern AI systems exhibit increasingly more advanced and efficient problem-solving in a wide range of domains, they appear, to us, limited in their ability to create and understand certain categories of high-order information such as matters of the human condition, societal dynamics, etc.

While the argument can be made that these systems are vastly different from us, it is also undeniable that they have also been exposed to an amount of information pertaining to these high-order concepts at a scale that no human has ever been, by far.

It thus appears to us that the wisdom and understanding required to make sense of these categories of information are not going to emerge meaningfully from the present paradigm.

### The need for higher fidelity human behaviour imitation

In the pursuit of designing systems that exhibit sapience, it appears to us that one has to position itself with respect to a gradient that spans from pure imitation of existing behaviour all the way to independent discovery.

From our perspective, sapience as it appears in humanity is an advanced product of evolution that sits at the top of an incredibly sophisticated dependency graph. As a result, we believe the chance of success in building sapient systems to drop very sharply for anything that distances itself from pure imitation of existing behaviour.

We thus believe that our best bet at embedding sapience into computational systems is through the betterment of human behaviour imitation modeling.

### Random search to better human behaviour imitation

#### Current AI paradigm

We observe that, over the past ~15 years, AI research has largely focused on exploring a wide range of priors within a gradient-based optimization dominated framework.

Gradient-based optimization methods are tightly coupled to the data distribution, funneling information from the data directly into the model’s representation space.

#### Random search as an extension

From our perspective there are three categories of spaces that problem-solving computational methods can live in: prior, data and random space.

—


The classical random space exploration method is random search. It is an optimization method that is employed when we 1) do not have the complete priors to successfully solve a task and 2) do not have a dataset to draw information from.

The classical random space exploration method, when we do have a dataset that we wish to draw information from, is evolutionary search. In evolutionary search, the representation space is also perturbed through random search, but data now plays a regularizing role, constraining random space exploration.

Inverse reinforcement learning is a class of methods that get to live in all three prior, data and random spaces. In practice however, the exploration of random space is minimal and focused on early-stage optimization.

—

Our hypothesis is that, given our non-omniscience, meaningful value remains unextracted when search and optimization are constrained to prior and data space.

#### Main constraints & remediations

From our perspective, evolutionary search permeates several conditions that, in practice, make their successful calibration quite exotic relative to modern gradient-based methods.

We introduce what we consider to be two of these conditions, and how we think best to approach them.

##### Data indirectness

Given that, relative to gradient-based methods, data is retrograded from being a direct to an indirect optimization signal, valuable data information now gets assimilated into representation space at a much slower and less efficient pace.

We propose to mitigate this phenomenon by largely building on top of the modern paradigm. In practice, this means first extracting information and rolling out policies with gradient-based methods, and secondly, within that generated realm, operate evolutionary search.

##### Navigating random space

Perturbing representation space using random space is a delicate endeavour. With respect to our objectives, much of random space is destructive noise.

We thus need to calibrate our search to satisfy a delicate balance between representation construction and noise application.

In the pursuit of that interest, we give our methods that have any impact on that balance the flexibility to both calibrate their noise application and shield their already constructed representations.

## Research strategy

We share, in this sub-section, our strategy for operationalizing our research endeavour.


### Motivation

Our belief is that, in practice, several conditions need to all be met in order to reach visible signs of deeper understanding and wisdom: imitation of human behaviour in its complete form, proper calibration of random search conditioned on gradient-based representations, etc.

In order to meet all of these conditions, we believe that a multitude of methods need to be incorporated into a working solution. Several of the methods that we later detail have, in isolation and in older forms, been scientifically peer-reviewed. However, many of them have not.

In order to account for time constraints, we are taking the (somewhat) measured “leap of faith” of building towards our objective while betting instead on our observed empirical results in addition to years of conceptual experimentation.

### Outline

We provide, in the upcoming `Specifications` section of this document 1) an appropriately detailed description of all the components envisioned for the implementation of both a) the envisioned approach & b) baselines to compare against; and 2) a proposed rough path for experimentation.

These components are meant to provide a solid framework for a sufficiently capable AI agent to both 1) generate the apparatus and 2) run extensive incremental experimentation in order to build any of the missing understanding that could be required in order to successfully combine all of these methods together.

### Current objective

At the present stage (2026/05), our research is geared towards attempting to first extract previously unextracted human behaviour characteristics out of human behaviour datasets.

# Specifications

This section is composed of three sub-sections: `Modeling methods`, `Data setup` & `Experimentation path`.

We specify in each of these sections our complete methodologies and a lot of reasoning behind our design choices.

## Modeling methods

We detail, in this sub-section, all of our `Evolutionary search` methods and how they concord with modern methods in `Leveraging the gradient-based paradigm`.


### Evolutionary search

We begin by introducing our methods that pertain to evolutionary search.


#### Core design

We begin by introducing 


##### Overview

There are many types of evolutionary search algorithms.


Our context is that of evolving neural networks (details in the `Neuroevolution` sub-section below).

In that context, there are roughly two practically proven classes of methods: genetic algorithms and evolution strategies.

Evolution strategies build on the idea of recombining agents, placing higher recombination weight to better performing agents.

We are of the opinion that recombination of neural networks, in practice, imposes two conditions that we believe too harmful for our purposes.

Firstly, recombination requires a network format suitable for recombination. In practice, this operation is typically applied to fixed connectivity networks in weight and bias space.

However, we believe fixed connectivity networks to be too prohibitive for our purposes.

The human behaviour that we attempt to imitate leverage very intricate network architectures.

While intelligence has been shown to emerge from human-made static architectures, sapience has not.

Secondly, we believe recombination to be too opinionated of a design decision.

It places its trust in that more valuable information will be gained than lost, and that the information that is lost is worth losing. While the first point has been shown to be true in practice, the second has not, for it has not been deemed an immediate concern by that research community.

We thus opt for a crossover-free genetic algorithm.


Our population maintains a single population of a fixed number `pop_size` of agents.

Each agent has three distinct roles: generator, discriminator and mutator (more details below).


The algorithm loops every iteration through three standard stages: mutation, evaluation and selection.

The generation of agents at iteration `<iter>` is `gen <iter>`.

###### Islands

We also choose to make the genetic algorithm island-based. We explain our reasoning in the `Adversarial imitation` section.

There are `num_islands` islands, each composed of the same `island_size` (`pop_size / num_islands`) number of agents.

###### Mutation stage

During the mutation stage, most agents (not all agents, see the upcoming `Selection stage` sub-section) are randomly perturbed.

We do not implement crossovers. Our perspective is that while there are many benefits to crossovers in isolated practical situations, they are very hard to set up properly in the more complex situation that we operate in.

Finally, as per their mutator role, all agents mutate themselves (we detail this operation in the `Mutator role` section).


###### Evaluation stage

During the evaluation stage, all agents behave in a virtual environment and are quantitatively graded with a fitness score.

###### Selection stage

Selection for both the islands and the population is achieved using 50% truncation selection.

This corresponds to ranking agents by fitness score, discarding the bottom 50% of agents and duplicating the top 50% of agents (to put it in illustrated language terms: selected agents become parents, produce two offspring, and erase themselves).

Per-island selection occurs every iteration. Every `island_merge_freq` iterations, the global population is ranked and 50% truncation selection occurs. New islands are then formed using the selected population at random.

As a result of per-island selection, every island has one agent with the highest fitness score. Its first of two offspring is not to be mutated during its upcoming mutation stage.

As a result of the population-wide selection that occurs every `island_merge_freq` iterations, only 

Finally, we choose 50% over smaller values to simply widen the search space as much as possible.

Since all selected parents produce two offspring each, we do not give preferential treatment to top parents. Our reasoning is two-fold: 1) if top parents came across relatively more valuable representations, their children are more likely to themselves become parents relative to other children with lower performing parents & 2) generational inheritance (described in the next sub-section).

#### Generational inheritance

Generational inheritance is meant to make evolutionary search more efficient with respect to evaluation.
It does so by having agents run on portions of a task instead of full tasks.


Agents that are selected transfer to their two offspring not only their genotype, but also their final environment state, final memory state, and cumulative lineage fitness scores.

During its own evaluation stage, the offspring thus loads and resumes from its parent’s memory state and environment state. Its fitness score becomes the sum of its lineage fitness score plus its own. The selection stage thus accounts for that sum rather than the individual’s fitness.

Agents are evaluated for `num_states` states. If the virtual environment terminates before `num_states` take place, we reset the environment and memory (but not the fitness score) and run for the remaining states.

—

This mechanism has an important impact on the selection process, as described in the previous sub-section. Children whose lineage was relatively more competent have more room to explore mutations that do not yield immediate returns.

We see value in this property for the same reasons that occur in the natural world.

#### Adversarial imitation

We optimize for imitation using adversarial evolution.

As mentioned previously, each agent is both a generator and a discriminator.

Every iteration, each agent is evaluated for both its generation and discrimination performance.

Generation is quite straightforward: the agent simply behaves in the virtual environment.

Discrimination is a bit more involved: the agent observes the behaviour of all other agents on its island.

—

We choose to operate adversarial imitation rather than supervised imitation.

Our reasoning is multi-fold:

1) Evolutionary adversarial imitation is quite open-ended up to the theoretical perfect discriminator / generator: new
opportunities to encode new representations always open up.

This is in contrast with a fixed supervised loss where representation creation is much more constrained to the path of optimization

2) The operationalization of adversarial imitation in an evolutionary search context is quite a bit different than it is with gradient-based optimization.

Adversarial imitation in gradient-based optimization appears to have two drawbacks that presently make them second-class citizens to diffusion models: 1) training instability and 2) one-shot generation.

Point 1 is only relevant in the context of gradient-based optimization and point 2 is only relevant in the context of having a fixed inference budget, which we do not (see the `Neural Networks` section).

We now introduce the method.




##### 
#### 

Every iteration, 

##### Application to generational inheritance

Generational inheritance is straight-forward to implement for the generator role.

However it is not so straightforward for the discriminator role, given that an agent’s chile will most likely not encounter the same generator 

#### Neural network

$$$$$ Evolution

Each agent maintains one neural network.

The network is a directed graph. Structurally, the network is made up of nodes and connections.

£££££ Computation 

While network evolution is performed per-network, computation is batched so it can run in parallel on the GPU.

The ‘3 x num_nodes’ 



















































#### Neural networks

Each agent maintains one neural network that is a directed graph of nodes.

There is no concept of layers in these networks.


##### Nodes

###### Overview

The graph has three types of nodes: `input`, `hidden` and `output` nodes.

Nodes can communicate through directed connections.

`input` nodes do not have `in_nodes`, whereas `hidden` and `output` nodes can have up to 3 (hard limit).
All nodes can have an unlimited number of `out_nodes`.

There are `n` `input` nodes, corresponding to the `n` real values that the network inputs. Input nodes thus only input their assigned network input.

Nodes do not assign weights to the values they receive. All nodes (including `input` nodes) run a standardization of the sum of their input values using Welford's online algorithm. They output the result of that standardization.

There are `m` `output` nodes, corresponding to the `m` real values that the network outputs.

—

A network is first initialized with its `n` `input` nodes and `m` `output` nodes and no connections.

Connections and `hidden` nodes are formed through the use of the `grow_node` mutation.

—

The `grow_node` mutation consists in creating a new `hidden` node and assigning it 2 `in_nodes` & 1 `out_node`.

We randomly sample the first `in_node` from the set of `receiving_nodes` (nodes that input at least one value).

We sample the second `in_node` from the set of `receiving_nodes` minus the first `in_node`.

This time however, instead of randomly sampling any of these nodes, we assign a preference to the nodes that are “near” the first `in_node`.

—

We maintain a `local_connectivity_probability` parameter 


During the `grow_node` mutation, we begin by selection 



#### Mutator role


### Leveraging the gradient-based paradigm

We now detail how we plan to bring evolutionary search and the modern paradigm together.



## Data setup

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

## Experimentation path



