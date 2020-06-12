# BSD-3-Clause License
#
# Copyright 2017 Orange
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
.. _pydcop_commands_generate_cdcop_random_bird:


pydcop generate continuous DCOP for random graphs
=====================

Continuous benchmark problem generator for random problems based on the bird function
---------------------------------

::

  pydcop generate cdcop_random_bird
                --agent_count <agent_count>
                --graph_density <graph_density>

Description
-----------

Generates a random C-DCOP based on the number of agents and the density of the graph for a specified relation.
The (binary) constraints are based on the relation function.

**Note:** the generated C-DCOP and distribution(s) are written to the standard output.
To write them in files,
you can use the ``--output <file>`` :ref:`global option<usage_cli_ref_options>`.


Options
-------

``--agent_count <agent_count>``
  Number of agents in the problem, must be >= 2.

``--graph_density <graph_density>``
  Density value of the graph, must be >= 0.1.

``--random_seed <random_seed>``
  The seed to be used in the generation of the random graphs

Examples
--------

Generate a C-DCOP representing a 4 agent continuous random problem, in extensive form::

    pydcop --output cdcop_random.yaml generate cdcop_random --agent_count "5" --graph_density "0.1"

"""
import logging
import random
import numpy as np
from collections import defaultdict
from os.path import splitext
import pathlib
from typing import Any, Dict, Tuple

import networkx as nx
import yaml
import importlib

from pydcop.dcop.dcop import DCOP
from pydcop.dcop.objects import (
    Variable,
    ContinuousDomain,
    AgentDef,
    ExternalVariable,
)
from pydcop.dcop.relations import (
    NAryMatrixRelation,
    Constraint,
    constraint_from_external_definition,
    NAryFunctionRelation,
    AsNAryFunctionRelation,
)
from pydcop.dcop.yamldcop import dcop_yaml

logger = logging.getLogger("pydcop.cli.generate")


def init_cli_parser(parent_parser):
    parser = parent_parser.add_parser(
        "cdcop_random_bird", help="Generate a continuous random benchmark problem based on the bird function"
    )
    parser.set_defaults(func=generate)

    parser.add_argument(
        "--agent_count", required=True, type=int, default=2, help="Number of agents in the problem"
    )
    parser.add_argument(
        "--graph_density", required=True, type=float, default=0.1, help="The density of the graph"
    )
    parser.add_argument(
        "--random_seed", required=False, type=int, default=1, help="The seed to be used in the generation of the random values"
    )

def generate(args) -> None:

    # Some extra checks on cli parameters!
    if args.agent_count <= 1:
        raise ValueError("--agent_count: The amount must be > 1")
    if args.graph_density < 0.1:
        raise ValueError("--graph_density: The value must be >= 0.1")

    # Check if the relation exists
    relation_path = pathlib.Path('tests/instances/bird_constraint.py')
    if not relation_path.is_file():
        raise ValueError(f"--relation: Could not find '{args.relation}'")

    # Create the problem
    dcop, var_mapping, fg_mapping = generate_continuous_random(
        relation_path.absolute(),
        args.agent_count,
        args.graph_density,
        args.random_seed,
    )

    # Define the graph
    graph = "constraints_graph"
    
    # Create the results dictionary
    output_file = args.output if args.output else "NA"
    dist_result = {
        "inputs": {
            "dist_algo": "NA",
            "dcop": output_file,
            "graph": graph,
            "algo": "NA",
        },
        "cost": None,
    }

    # Create the output
    if args.output:
        with open(output_file, encoding="utf-8", mode="w") as fo:
            fo.write(dcop_yaml(dcop))
        path, ext = splitext(output_file)
        # if args.var_dist:
        dist_result["distribution"] = var_mapping
        dist_output_file = f"{path}_vardist{ext}"
        with open(dist_output_file, encoding="utf-8", mode="w") as fo:
            fo.write(yaml.dump(dist_result))

    else:
        print(dcop_yaml(dcop))
        dist_result["distribution"] = fg_mapping
        print(yaml.dump(dist_result))

def generate_continuous_random(
    relation: str,
    agent_count: int,
    graph_density: float,
    lower_bound: float=0.0,
    upper_bound: float=1.0,
    random_seed: int=1,
) -> Tuple[DCOP, Dict, Dict]:

    # Set the seed
    np.random.seed(random_seed)

    # Create the randomized graph
    # https://networkx.github.io/documentation/stable/_downloads/networkx_reference.pdf
    # n(int) – The number of nodes.
    # p(float) – Probability for edge creation.
    # seed(integer, random_state, or None (default)) – Indicator of random number generationstate
    graph = nx.generators.random_graphs.fast_gnp_random_graph(
        n=agent_count,
        p=graph_density,
        seed=random_seed,
    )

    # Check the graph
    # All nodes have at least one edge
    neighbors_list = [len(list(nx.neighbors(graph, i))) for i in graph.nodes()]
    most_neighbors = np.argmax(neighbors_list)
    for node_index, number_of_neighbors in enumerate(neighbors_list):
        if number_of_neighbors == 0 and node_index != most_neighbors:
            graph.add_edge(node_index, most_neighbors)

    # Create the domain
    domain = ContinuousDomain(
        name="bounded", 
        domain_type="continuous", 
        lower_bound=lower_bound, 
        upper_bound=upper_bound,
    )

    # Create the variables based on the graph and the domain (equal for all)
    variables = generate_variables(graph, domain)

    # Create the constraints (as functions)
    constraints = {}
    binary_constraints, external_constants, external_domains = generate_constraints(
        relation, graph, variables
    )
    constraints.update(binary_constraints)

    all_domains = [domain]
    all_domains.extend(external_domains)

    # Create the agents
    agents = {}
    fg_mapping = defaultdict(lambda: [])
    var_mapping = defaultdict(lambda: [])
    for index in graph.nodes:
        agent = AgentDef(f"a_{index}")
        agents[agent.name] = agent
        var_mapping[agent.name].append(f"v_{index}")

    # Create the DCOP
    name = f"cdcop_random_bird_{agent_count}_{graph_density}"
    dcop = DCOP(
        name,
        objective='max',
        domains={d.name: d for d in all_domains},
        variables={v.name: v for v in variables.values()},
        agents=agents,
        constraints=constraints,
    )
    dcop.external_variables = external_constants

    return dcop, dict(var_mapping), dict(fg_mapping)

def generate_variables(grid_graph: nx.Graph, domain: ContinuousDomain):
    variables = {}
    for index in grid_graph.nodes:
        variable = Variable(f"v_{index}", domain)
        variables[variable.name] = variable
    return variables

def generate_constraints(relation: str, grid_graph: nx.Graph, variables: Dict[str, Any]) -> Dict[str, Constraint]:
    # Create the buffer for the constraints
    constraints: Dict[str, Constraint] = {}
    external_constants = {}
    external_domains = []

    # Loop over the edges of the graph
    for nodes in grid_graph.edges:
        # Select two nodes of the graph
        (r1, r2) = sorted(nodes)
        v1_name = f"v_{r1}"
        v2_name = f"v_{r2}"

        # Create variables
        v1 = variables[v1_name]
        v2 = variables[v2_name]

        # Create the constraint
        constraint = constraint_from_external_definition(
            name=f"bird_{v1_name}_{v2_name}",
            source_file=relation, 
            expression=f"source.bird_normalized_inputs({v1_name}, {v2_name})",
            all_variables=[v1, v2],
            properties={'Lipschitz_constant':1326.5 * (4.0 * np.pi)},
        )

        # Add the constraint to the dictionary
        constraints[constraint.name] = constraint

        # Create a dict of the external_constants
        external_domains = [c.domain for c in external_constants.values()]
    return constraints, external_constants, external_domains


