# artificial_wisdom_research

===

This research codebase is an attempt to bring to light various novel results emanating from an interplay of hypotheses that we describe below.

We have generated several peer-reviewed findings that relate to these hypotheses. They are available to consult at `articles/`.

—

In an ideal world, each of these described hypotheses ought to further be carefully examined, both independently and in incremental relation to each other, before attempting such a venture.

We are however short on time and resources. We are thus taking a “leap of faith” and placing my trust on the many years of thought experiments that we have done on these hypotheses, hoping that it can fill that experimentation gap.

===


Modern AI systems are conceived using only a small subset of all computational methods.

For instance, while the realm of computational search/optimization methods is vast, we only find gradient-based optimization methods in the aforementioned systems.

We can thus deduce that value has been more readily attainable in focusing on this narrow subset of techniques.

The research community is however uncertain about the value reach that the current research efforts are to yield.

We hypothesize that various pockets of value are hardly reachable.

—

There is, in gradient-based optimization, a very tight coupling to the data distribution.

In that paradigm, computational information of the data distribution is funneled directly into the models’ representation space.

Our core hypothesis is that there is untapped value in leaving the search/optimization room to explore beyond the confines of data space.

This hypothesis is rooted in the real world observation that creativity often emerges from unpopular/unconventional trajectories.

Our best bet to execute this vision are evolutionary algorithms.

In this paradigm, data is retrograded to regularizing and the representation space is now perturbed using random search.

—

X Relative downsides of evolutionary algorithms

Because data is now indirect with respect to representation space formation, data-originating information is encoded into the representation space at a much slower pace.

Realistically, this means that we should extract all of the information available through gradient-based methods so that we do not need to do it with evolution

X Orchestrating evolutionary algorithms is quite different from orchestrating gradient-based methods.

Be

X Opportunity to combine gradient-based methods and evolutionary algorithms


—


X Generative Adversarial Evolution + Network sharing vs Action prediction

Paper 2

Several key differences:
Instead of having two separate populations of generator and discriminator agents, we now have a single population of agents that are both generator and discriminator. Their model is shared, including the output space (meaning that when generating, an agent has access to it discrimination output and vice versa)
Instead of a single agent match per iteration, we now run k matches. The default k=3.

X Neural networks of dynamic complexity

X All representational capacity in the architecture

Network connections do not have weights. Instead, 
they simply sum the outputs of each neuron that they input, feed it through a per-neuron running standardization and output that transformed signal

Each neuron computes a sum of all 

X Generational inheritance

Paper 3 extract phrasing

In the context of evolutionary adversarial generation, generator memory is straightforward to 

X Evolved network on top of gradient-based network

Modern AI policy network
Then have a network evolve akin to a parasite.
Starts by mapping from the output space back into the output space
Can start to go fetch from earlier layers after a certain point

X Mutator role

Agents are now also able to mutate themselves. In order to do so, they input from network

X Objective: show that evolutionary algorithms can extract untapped value on top of gradient-based methods

	X Attempt showing in the context of static policy behaviour imitation, however it is likely
	that there is too little value left to extract.

	We have a dataset of human 

