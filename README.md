This is a single-page, carefully sectioned markdown document that details our research at various levels of abstraction.

It is optimized for both human and AI agent readability.

The tipping point motivation for creating this document was to fully decouple research methodology and experimentation away from a consistent codebase, which has only recently become legitimate.

This document has two main sections: `# Overview` and `# Specifications`.

We describe, in `# Overview`, our high-level understanding and research reasoning so as to contextualize `# Specifications`.

# Overview

We begin this section by sharing, in `## Why artificial sapience?` our current understanding of both sapience and its relation with artificial intelligence.

We then proceed, in `## Central hypotheses`, to introduce and motivate several hypotheses that shape the nature of our research direction.

Finally, in `## Research strategy`, we share insights into how we operationalize our research endeavour.

## Why artificial sapience?

We motivate, in this sub-section, the value of artificial sapience and its contrast with artificial intelligence.


### Sapience

Intelligence, wisdom and understanding are three advanced concepts that concern themselves with problem solving.

Just like most high-order concepts, all these concepts have neither universally agreed definitions nor clear boundaries with each other.

To the best of our knowledge however, sapience is the term that best englobes the three terms in how they manifest in human beings.

### Intelligence

Our perspective is that intelligence is proportional to the likelihood of solving a specific set of problems.

There is per-problem intelligence: an entity can be very good at solving very specific problems but very bad at solving everything else. In that case, it is narrowly very intelligent.

More broadly valuable however is general intelligence. Human beings and modern artificial intelligence systems for instance are able to solve a wide range of problems.

### Sacrificing sapience for the sake of greater intelligence

In the pursuit of increasing the likelihood of solving a given set of problems, greater intelligence is willing to forgo certain endeavours, such as 1) considering information pertaining to the greater context that the set of problems is in (wisdom) & 2) building comprehension that is not strictly required to “solve” the tasks in their current specification (understanding).


### The artificial intelligence imbalance

We believe that there is an imbalance in the world of artificial systems wherein intelligence is prioritized at the expense of the other concepts that make up sapience.

This repository represents our attempt at a contribution in the pursuit of restoring this balance.

## Central hypotheses

We share, in this sub-section, the three core hypotheses that motivate our research endeavours.


### Modern AI systems, wisdom & understanding

While modern AI systems exhibit increasingly more advanced and efficient problem-solving in a wide range of domains, they appear, to us, limited in their ability to create and understand certain categories of high-order information such as matters of societal dynamics, the human condition, etc.

While the argument can be made that these systems are vastly different from us, it is also undeniable that they have also been exposed to an amount of information pertaining to these high-order concepts at a scale that no human has ever been, by far.

It thus appears to us that the wisdom and understanding required to make sense of these high-order information categories are not going to emerge meaningfully from the present paradigm.

### The need for higher fidelity human behaviour imitation

In the pursuit of designing systems that exhibit sapience, it appears to us that one has to position itself with respect to a gradient that spans from pure imitation of existing behaviour all the way to independent discovery.

To the best of our knowledge, sapience as it appears in humanity sits at the top of an incredibly sophisticated dependency graph that was built over the course of billions of years of universe-scale evolutionary pressure. As a result, we believe the practical chances of success in building sapient systems to drop very sharply as we ever so slightly distance ourselves from pure imitation of existing behaviour.

We thus believe that our best bet at embedding sapience into computational systems is through the betterment of human behaviour imitation modeling, through both 1) higher fidelity human behaviour datasets & 2) methods that can better model these higher fidelity datasets.

### Improving human behaviour imitation through random space exploration

#### Current AI paradigm

We observe that, over the past ~15 years, AI research has largely focused on exploring a wide range of priors within a gradient-based optimization dominated framework.

Gradient-based optimization methods are tightly coupled to the signal that they learn from (data or objective), funneling information from these signals directly into a model’s representation space.

#### Why random space exploration

We see three types of spaces that problem-solving computational methods can navigate: prior, signal and random space.

Each space has different degrees of both theoretical bounds and ever-changing practical bounds.

Our perspective boils down to the opinion that it is possible, and relatively most worthwhile, for us to push the practical bounds of random space exploration.

These beliefs are rooted in our own lived experience, and the parallels that we observe between it and the problem-solving computational world, which are both out of scope in the context of this document.


#### Random space exploration methods

The classical random space exploration method is random search. It is an optimization method that is employed when we both 1) cannot build from signal space information and 2) do not have all the necessary priors to tackle a problem.

The classical random space exploration method, when we can build from signal space information, is evolutionary search. In evolutionary search, the representation space is also perturbed through random search, but the signal now plays a regularizing role, constraining further random space exploration.

Inverse reinforcement learning is a class of methods that get to live in all three prior, signal and random spaces. In practice however, we personally see its exploration of random space as a means to better explore signal space.


We envision a path to expand the practical bounds of random space through evolutionary search.

#### Main constraints & remediations

From our perspective, evolutionary search permeates several conditions that, in practice, make their successful calibration quite exotic relative to modern gradient-based methods.

We introduce what we consider to be two of these conditions, and how we think best to approach them.

##### Signal indirectness

Given that, relative to gradient-based methods, signal is retrograded from being a direct to an indirect optimization cue, valuable signal information now gets assimilated into representation space at a much slower and less efficient pace.

We propose to mitigate this phenomenon by largely building on top of the modern AI paradigm. In practice, this means first extracting information and training policies with gradient-based methods, and only then, within that realm, operate evolutionary search.

##### Navigating random space

Perturbing representation space using random space is a delicate endeavour. As the solution space gets smaller, an ever greater portion of random space can be considered destructive noise.

We thus need to calibrate our search to satisfy a delicate balance between representation construction and noise application.

In the pursuit of that interest, we implement mechanisms throughout our methods that allow for 1) the calibration of noise application and 2) the shielding of valued representations.

## Research strategy

We share, in this sub-section, our strategy for operationalizing our research endeavour.


### Motivation

Our belief is that, in practice, several conditions need to all be met in order to reach visible signs of sapience: imitation of human behaviour in its most native form, proper calibration of random search conditioned on gradient-based representations, etc.

In order to meet all of these conditions, we believe that a multitude of method iterations / novel methods need to be incorporated into a working solution.

In this attempt of ours, several of the methods that we later detail have, in isolation and in older forms, been scientifically peer-reviewed. However, many of them have not.

In order to account for time constraints, we are taking the (somewhat) measured “leap of faith” of building towards our objective while betting instead on 1) our observed empirical results and 2) a large body of conceptual experimentation.

### Outline

We provide, in the upcoming `# Specifications` section of this document 1) an appropriately detailed description of all the components envisioned for the implementation of both a) the envisioned approach & b) baselines to compare against; and 2) a proposed rough path for experimentation.

These components are meant to provide a solid framework for a sufficiently capable AI agent to both 1) generate the research apparatus and 2) run extensive incremental experimentation in order to build any of the missing understanding required in order to successfully combine all of these methods together.

### Current objective

At the present stage (2026/05), our research is geared towards attempting to show that our methods can achieve more “perfect” imitation than existing methods.

We propose a ramp up strategy where we explore the imitation abilities of both baseline and our methods on increasingly complex behaviour, on increasingly complex tasks.

# Specifications

This section is composed of three sub-sections: `## Modeling techniques`, `## Data setup` & `## Experimentation path`.

We specify in each of these sections our complete methodologies and the reasoning behind each of our design choices.

## Modeling techniques

We detail, in this sub-section, an interplay between evolutionary search and the current gradient-based paradigm.

In order to facilitate assimilation, we begin by describing, in `### Self-supporting evolutionary search` our methods that do need to be understood in the context of the gradient-based paradigm.

In `### Leveraging the gradient-based paradigm`, we then explain how we tie in the whole approach together.


### Self-supporting evolutionary search

We begin by introducing our self-sufficient evolutionary search methods.

In `#### Core design`, we detail design decisions that pertain to the core of the evolutionary search.

In the later sub-sections we detail more specific mechanisms with the following purposes:
- `#### Neural networking`: flexible random and agentic representation construction
- `#### Generational inheritance`: increasing evaluation efficiency and expanding exploration
- `#### Adversarial imitation optimization`: open-ended behaviour imitation


#### Core design

There are many types of evolutionary search algorithms that offer a wide-range of trade-offs that suit different contexts.

In our context of evolving neural networks (details in `#### Neural networking`), there are roughly two practically proven classes of methods: genetic algorithms and evolution strategies.

We opt for genetic algorithms.

##### Why not evolution strategies?

Evolution strategies methods build on the idea of population-scale recombination of agents, placing higher recombination weight on better performing agents.

We are of the opinion that recombination of neural networks, in practice, imposes two conditions that we believe make them unsuitable for our purposes.

###### Network design requirements


Firstly, recombination requires a network format amenable for recombination. While methods like NEAT have shown that dynamic-connectivity networks can be recombined, these methods bake in a wide range of assumptions and accept many practical inefficiencies. In practice, recombination therefore seems to require the use of fixed-connectivity networks, so as to operate in real value parameter (weight and bias) space. 

While general intelligence has emerged from human-made fixed connectivity networks, it has thus far only emerged in a pure gradient-based optimization context.

It is thus still unclear, given the many disconnects between gradient-based optimization and evolutionary search, how much value can be leveraged from fixed-size networks.

We further explain our reasoning for the use of dynamic-connectivity networks in `#### Neural networking`.

###### Representation incompatibility


Secondly, we believe recombination to be too opinionated of a design decision.

Recombination assumes that the gains from combining partial solutions outweigh the disruption caused by merging incompatible representations. While this tradeoff often works empirically, the degree to which useful information is destroyed remains an active research concern, given the now large disconnect with gradient-based methods in valuable applicability.

##### Genetic algorithm

Following our line of reasoning about recombinations, we opt to make our genetic algorithm crossover-free. However, we detail in `#### An opportunity for indirect recombination` how we create a channel for agents to observe and integrate other agents’ valuable representations, thus not completely forgoing the value of recombination.

—

Our population maintains a single population of a fixed number `pop_size` of agents.


The algorithm loops every iteration through three standard stages: mutation, evaluation and selection.

The generation of agents at iteration `<iter>` is `gen <iter>`.

##### Islands

We also choose to make the genetic algorithm island-based. We explain our reasoning in `##### Operationalization of generational inheritance`.

There are `num_islands` (1+) islands, each composed of the same `island_size` (`pop_size / num_islands`) number of agents.

##### Mutation stage

During the mutation stage, all but a few agents (see `###### Elitism`) undergo mutation.

The mutation stage is fully tied to the agents’ neural network systems. We thus leave out the details to that particular sub-section (see `#### Neural networking`).

##### Evaluation stage

During the evaluation stage, all agents behave in a virtual environment and are assigned a quantitative personal fitness score.

The mechanics that decide the value of that personal fitness score are detailed in `#### Adversarial imitation`.

* In order to better understand the notion of “personal fitness score”, we quickly foreshadow here the concept of “lineage fitness score”, with an agent fitness score being the sum of both its personal and lineage fitness score (details in `#### Generational inheritance`).

##### Selection stage

We now introduce our selection stage heuristics.


###### Truncation

Selection for both the islands and the population is achieved using the standard truncation selection method with a rate of 50%.


This corresponds to ranking agents by fitness score, discarding the bottom 50% of agents and duplicating the top 50% of agents (to put it in illustrated language terms: selected agents become parents, produce two offspring identical to themselves, and then erase themselves).

###### Why 50%?

50% is the largest truncation selection percentage wherein we do not need heuristics to decide how many offspring a parent produces (for instance, 60% needs a heuristic to decide which parents produce how many of the remaining 40% of agents).

Smaller heuristic-free truncation selection percentages like 25%, 12.5% choose to increasingly diminish random space exploration breadth in order to increasingly invest resources in the best current solutions.

We choose 50% because 1) we believe the optimization dynamics of our `#### Adversarial imitation` context to be too intricate to confidently make the type of trade-off described above & 2) because we do not see value in further exploring optimizing the number of offspring per parent: our reasoning is that if top parents came across relatively more valuable representations, their children are more likely to themselves become parents relative to other children with lower performing parents.

###### Interaction between islands and the global population

Per-island 50% truncation selection occurs every iteration except every `island_merge_freq` iterations (unless we run a single island).

Every `island_merge_freq` iterations, the global population is ranked and a global 50% truncation selection occurs instead.

Islands are then reformed by dispatching the global population’s new offsprings randomly across the `num_islands` islands.


###### Elitism

Following the evaluation stage, every island has an elite: the agent with the highest fitness score on the island.

The elite is marked to have one of its two offspring not be mutated during its next iteration mutation stage.

This mark remains applied for either per-island selection and population-wide selection (though population-wide selection could see a given island’s elite agent discarded due to not being ranked in the top 50% fitness scores of the global population).

###### Why elitism?

We choose to implement some form of elitism, over no form of elitism, in order to provide some shielding against particularly rough random space regions where representational improvements are hardly made.

However we decide not to implement a heavier form of elitism so as to not considerably slow-down the search: the representations that we are looking for are very complex and are very deep in the search space.

#### Generational inheritance

Generational inheritance is meant to make evolutionary search more efficient with respect to evaluation.

It does so by having agents run on portions of a task instead of full tasks. For our purposes, it means having individual agents operate on sub-sections of human behaviour


Agents that are selected transfer to their two offspring not only their genotype, but also their final environment state, final memory state, and cumulative lineage fitness scores.

During its own evaluation stage, the offspring thus loads and resumes from its parent’s memory state and environment state. Its fitness score becomes the sum of its lineage fitness score plus its own individual fitness score. The selection stage then accounts for the sum of the lineage and individual fitness scores.

Agents are evaluated for `num_states` states. If the virtual environment terminates before `num_states` take place, we reset the environment and memory (but not the fitness score) and run for the remaining states.

—

This mechanism also has an important impact on the selection process, as described in the previous sub-section. Children whose lineage was relatively more competent have more room to explore mutations that do not yield immediate returns.

We see value in this property for the same reasons that occur in the natural world.

#### Adversarial imitation optimization

In its most basic form, adversarial imitation optimization is a method where a generator produces some artifact and a discriminator attempts to dissociate generated and real artefacts.

The generator is optimized to produce artefacts that fool the discriminator, while the discriminator is optimized to better dissociate.

##### Why adversarial imitation?

We choose to operate adversarial imitation over other forms of supervised/unsupervised imitation for the following reasons.

###### Open-ended optimization trajectory

Most non-adversarial imitation methods optimize against a relatively fixed objective. Even when expressive models are used, representation formation is ultimately constrained by a static loss function defined over a fixed target distribution.

Adversarial imitation differs in that the optimization target is itself adaptive.

The generator continuously optimizes to produce behaviour that cannot be distinguished from real behaviour, while the discriminator continuously optimizes to better dissociate generated and real behavioural artefacts. As each side improves, the effective optimization landscape changes.

This produces a fundamentally more open-ended optimization process.

Rather than following a single relatively static optimization path, adversarial imitation continuously creates new generative and discriminative pressures, allowing new behavioural and representational strategies to emerge throughout optimization.

In a population-based setting, this effect compounds further.

Rather than a single generator–discriminator pair, many generators and discriminators interact simultaneously, creating a heterogeneous and continually shifting set of selective pressures. Multiple behavioural equilibria, discriminative strategies, and representational niches may therefore coexist and compete, further reducing pressure toward convergence on a singular behavioural solution.

###### Sidestepping the constraints of gradient-based adversarial optimization

Adversarial optimization in gradient-based optimization is commonly associated with two major drawbacks: a) training instability & b) fixed inference budget.

Both drawbacks do not pertain to our conditions: 1) training instability arises from generator–discriminator gradient dynamics which are not pertinent to evolutionary search & 2) our methods have an inherently dynamic inference budget (see `###### num_node_passes_per_input`).

##### Core design

We have one population.

Every agent in the population is both a generator and a discriminator.

During the evaluation stage, the following occurs:
1) Every agent generates behaviour
2) Every agent discriminates the behaviour of
    a) the `island_size` agents on its island (including its own behaviour)
    b) `island_size` human subjects (we do so to make sure the discriminator does not successfully build representations that leverage distribution imbalance)


###### Network architecture ramification


Our central course of action in the pursuit of higher fidelity behaviour imitation is the use of a form of population-based adversarial imitation optimization, wherein every agent in our population both generates behaviour and discriminates the behaviour of all other agents on its island.

##### Operationalization of generational inheritance

Generational inheritance is straight-forward to implement for the generator role: a given generator begins an episode/task, runs for `num_states`, and if selected, has its children resume from where it left off.

Without the use of islands, generational inheritance is however not straightforward to use for the discriminator role. The reason is that we are faced with a trade-off where if we want discriminator agents to be able to consistently transfer to their offspring, we need to 
- 
-
For the discriminator role, 




However it is not so straightforward for the discriminator role, given that an agent’s chile will most likely not encounter the same generator 

#### Neural networking

We now describe our neural networking system.

It is the product of a large dependency graph of design decisions. We thus propose to begin by fully describing the design and later motivate each of the design decisions.

##### Design

Our neural networking system can be decomposed into two entities: a behaviour network and a mutator network.

Both networks are composed of nodes that share information through connections.

There is no concept of layers in these networks: all nodes run in parallel and all connections take one timestep to propagate forward.

The networks’ information unit is singular real values.

Nodes in these networks do not maintain weights nor biases.
However, each node, except `conditioning` and `mutation signal` nodes, maintains parameters enabling a running standardization of a function of their input values.

The representational capacity is thus carried by both network architectural pathways and node standardization parameters.

There are 3 categories of nodes: 1) `input`, 2) `action` and 3) `intermediate`.

—


The three types of input nodes are 1) `observation`, 2) `conditioning` and 3) `mutation signal` nodes.

`observation` nodes process and make available observations from the virtual environment (e.g. engine state, gradient-based representations).

`conditioning` nodes process and make available conditional information (e.g. session number, subject id).

`mutation signal` nodes emit several types of signals for use by the mutator network. We detail these signals in `### XXXX`.

—

The two types of action nodes are 1) `behaviour` and 2) `discrimination` nodes.

`action` nodes are used in two ways: to generate actions and to process other agents’ actions.
When `behaviour` nodes are used to generate actions, `discrimination` nodes are used to process other agents’ actions and vice versa.


`behaviour nodes` process unit-decisions (e.g. rotor actuation, high-level signals) while the `discrimination` node processes single-values as per `#### Adversarial imitation optimization`.

—

Finally, the two types of `intermediate` nodes are 1) `hidden` and 2) `mutating` nodes.

`hidden` nodes are specific to the behaviour network while `mutating` nodes are specific to the mutator network.

While all other nodes are created at initialization and made permanent, these two types of nodes are grown and pruned throughout the networks’ lifecycle.

xxxxxxx

Each node, except input nodes, maintains a list of `in` nodes: nodes that connect to that given node.
Each node has up to 2 `in` nodes.


####### Grow node

Three nodes are sampled (details in `##### XXX`).

###### The mutator network

The mutator network is given the full responsibility of mutating both the behaviour network and itself.


####### `mutation signal` nodes

In order to guide this process, the mutator network has access to information through the `mutation signal` nodes.

There are X `mutation signal` nodes:

1) The `grow/prune` node: outputs `1` to indicate a `grow node` mutation signal and `-1` to indicate a `prune node` mutation signal

2) The `behaviour/mutator` node: outputs `1` to indicate mutation on the behaviour network and `-1` to indicate mutation on itself.

3) The `node sampling` node: outputs `1` to indicate the beginning of a node sampling process and `-1` the rest of the time. During the `grow node` mutation it thus outputs `1` at 3 se
1) grow/prune, 2) behaviour/mutator, 3) 


####### 






##### Motivation

We have, in `### Improving human behaviour imitation through random space exploration`, esteemed a significant disconnect between evolutionary search and gradient-based optimization.

It has however been shown that evolution search can be used in order to optimize neural networks conceived for the purpose of gradient-based optimization. 

We argue in the following sub-sections that these networks are not unsuited but rather ill-suited to leverage random space exploration. We introduce in parallel what we believe to be better suited alternatives.

###### On random perturbations

Random perturbations are the essence of random space exploration.

A random perturbation is the application, on top of an existing construction, of a unit randomly sampled from a practically bounded search sub-space.

We argue that the value yield of successive random perturbations as they occur in evolutionary search is a reflection of 1) the overlap between the search space and the solution space and 2) the isolated value yield of single random perturbation events.

Our methods put a strong emphasis on point #2.


## Why a mutator network?

Setting up a mutator network in the manner that we do corresponds to making the assumption that: it is possible, in a given realm, to draw and build on top of a given amount and quality of information such that we can yield further value than through random perturbations alone.

We believe this assumption to be increasingly true the more complex the behaviour we adversarially imitate, by virtue of greater open-endedness. 




##### Parameters

###### Overview

Each network maintains its own set of these 4 mutable parameters:
- `avg_num_grow_mutations`, float, x >= 0
- `avg_num_prune_mutations`, float, x >= 0
- `num_node_passes_per_input`, int, x >= 1
- `local_connectivity_probability`, float, x >= 0, x <= 1

We decide to make these parameters mutable because we do not believe to have good general priors to set them properly ourselves.


###### `self.avg_num_grow_mutations` & `self.avg_num_prune_mutations`

The `avg_num_grow_mutations` and `avg_num_prune_mutations` values control how much growing and pruning randomly occurs.

We create these parameters in order to give the evolutionary search the flexibility to control the average occurrence of both mutations.

We believe this to be valuable because different stages of the evolution process likely benefit from different growth/prune regimes.

—

We randomly set the initial value, independently for both `avg_num_grow_mutations` `avg_num_prune_mutations`, to a random value between 0.1 and 1.0 (uniform sampling).

This wide range reflects the fact that we do not have good priors for setting these values.
Because we setup the perturbation of these parameters to be multiplicative (see ###### `self.perturb_parameters()`). We set the lower bound to 0.1 so as to not be trapped by being too close to 0, and the upper bound to 1.0 so as to not risk too much representation perturbation in early evolutionary search.


###### `self.num_node_passes_per_input`

The `num_node_passes_per_input` value controls how many node passes the network operates for every new series of input values. When it is greater than one, the given series of input values is simply fed repeatedly. Only the final output values are fed out of the network.

We create this parameter in order to give networks the flexibility to control how much processing time a given series of input values is given before moving on to a new one.

We believe this to be valuable for two main reasons:

1) If the network grows large, given that our networks do not use layers, an increasing number of node passes will be required in order for a given node to communicate to another given node.

2) We do not want to get in the way of the possibility that certain network configurations could derive value from having more network structure involved for any series of input values.

—

We randomly set the initial value of this parameter to a random value according to a shifted geometric distribution where p(1) = 50%, p(2) = 25% etc.

We choose this format because the evolutionary search begins with minimal networks and large `num_node_passes_per_input` are likely to be a waste of compute. However we do not have good priors for this design decision and thus wish to leave the option for large values open.

###### `self.local_connectivity_probability`


The `local_connectivity_probability` value controls the probability of accepting the current set of nodes considered (see `###### self.grow_connection()`). When `local_connectivity_probability` is high, nearby nodes are more likely to be made connected to each other.

We create this parameter in order to give the evolutionary search the flexibility to have some control over whether to push for local or global connectivity at different stages of the search. We choose to make this a global per-network parameter rather than a per-node parameter in order not to explode the search dimensionality.

—

We randomly set the initial value of this parameter to a value between 0 and 1 (uniform sampling).
This reflects, once again, that we do not know what a good initial value is.

##### Mutation stage

All networks are modified during the mutation stage.

The modifications made during the mutation stage are both random or agentic (see `#### Mutator role`).

There are three sub-stages in the random mutation stage: perturbing parameters, pruning node(s) and growing node(s). Parameter perturbation occurs first so that node growing and pruning reflect that change. Node pruning occurs after node growing so as to not prune untested new representational capacity. 

###### `self.perturb_parameters()`

####### `self.avg_num_grow_mutations` & `self.avg_num_prune_mutations`

rand_val = 1.0 + 0.01 * random gaussian sample (independent for both parameters)
parameter *= rand_val

—

This random mutation is multiplicative and makes going from 10 to 11 as likely as going from 1.0 to 1.1.
We choose multiplicativity over additivity in order to account for X.

We handpick the 0.01 sigma value somewhat randomly.

####### `self.num_node_passes_per_input`

rand_val = uniform sampling of 1, 0 or -1
parameter += rand_val (while making sure parameter stays in its desired range)

—

We believe that larger and deeper networks are likely to benefit from running multiple node passes. We thus purposely make this parameter able to vary quite a bit over time.

We do not want a network to overfit to its current `self.num_node_passes_per_input` value.

####### `local_connectivity_probability`

rand_val = 0.01 * random gaussian sample
parameter += rand_val (while making sure parameter stays in its desired range)

—

X

###### Grow node(s)

`self.grow_node()` is the only method that creates more representational capacity in a given network.

It does so by sampling three existing nodes and creating a `hidden` node that inputs from the first two sampled nodes and outputs to the third sampled node.

####### Sampling nearby nodes

####### Grow connection

###### Prune node(s)

####### Prune connection


##### Self mutability

We now detail how agents’ networks can evolve to alter their own connectivity.

We have the base network that we have mentioned until now.

And now we have a second network which we call the mutating network that grafts itself on top of the base network.

This mutating network grows only by random perturbation in the same fashion that the base network is randomly perturbed.

The mutating network grows mutating nodes that are like the base network’s hidden nodes except that:
- they always out

The mutating network 


##### Indirect recombination









### Leveraging the gradient-based paradigm

We now detail how we plan to bring evolutionary search and the modern paradigm together.

Grafting network evolution on top of fully functional gradient-based models.




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

We now detail a sequential order of experimentation and their purposes.

We propose to first attempt our most complex experiment using our full list of methods and most advanced dataset, and only if it fails do we start to remove some of the complexity. 

Our most complex experiment is that of 
