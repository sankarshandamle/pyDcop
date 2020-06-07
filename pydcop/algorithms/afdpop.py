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

AF-DPOP: Approximate Functional Dynamic Programming Optimization Protocol
-----------------------------------------------

Approximate Functional Dynamic Programming Optimization Protocol is an approximate,
inference-based, dcop algorithm implementing a dynamic programming procedure
in a distributed way as described in:

`New Algorithms for Functional Distributed Constraint Optimization Problems, Hoang2019 (http://arxiv.org/abs/1905.13275)`

AF-DPOP works on a Pseudo-tree, which can be built using the
:ref:`distribute<pydcop_commands_distribute>` command
(and is automatically built when using the :ref:`solve<pydcop_commands_solve>` command).

This algorithm has no parameter.


Example
^^^^^^^
::

    pydcop -algo afdpop tests/instances/cdcop_branin_func.yaml


"""
import os
import sys
import time

from random import random
from typing import Dict, Iterable, Any, Tuple, Callable, List, Union
import pickle

import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator

from pydcop.utils.simple_repr import SimpleRepr
from pydcop.computations_graph.pseudotree import get_dfs_relations
from pydcop.infrastructure.computations import Message, VariableComputation, register
from pydcop.dcop.relations import (
    RelationProtocol,
    NAryMatrixRelation,
    NAryFunctionRelation,
    constraint_from_str,
    Constraint,
    find_arg_optimal,
    join,
    projection,
    generate_assignment_as_dict, 
    filter_assignment_dict, 
    AbstractBaseRelation,
)
from pydcop.algorithms import ALGO_STOP, ALGO_CONTINUE, ComputationDef, AlgoParameterDef

from pydcop.dcop.objects import (
    ContinuousDomain, 
    Domain, 
    Variable,
)

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html#scipy.optimize.dual_annealing
# maxiter=1000 (default)
from scipy import optimize
# https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
from scipy.interpolate import interp1d, LinearNDInterpolator

GRAPH_TYPE = "pseudotree"

algo_params = [
    # Number of discrete points to used per domain
    AlgoParameterDef("number_of_discrete_points", "int", None, 3),
    # Number of iterations for the update of the values
    AlgoParameterDef("number_of_iterations", "int", None, 100),
    # Threshold (for the absolute difference of the value step) used for termination 
    AlgoParameterDef("threshold", "float", None, 1e-3),
    # Learning rate of the algorithm (during update of the values of the domains)
    AlgoParameterDef("alpha", "float", None, 0.01),
]

def build_computation(comp_def: ComputationDef):
    computation = AfDpopAlgo(comp_def)
    return computation


def computation_memory(*args):
    raise NotImplementedError("AF-DPOP has no computation memory implementation (yet)")


def communication_load(*args):
    raise NotImplementedError("AF-DPOP has no communication_load implementation (yet)")


class AfDpopMessage(Message):
    def __init__(self, msg_type, content):
        super(AfDpopMessage, self).__init__(msg_type, content)

    @property
    def size(self):
        if self.type == "UTIL":
            # UTIL messages are multi-dimensional matrices
            shape = self.content.shape
            size = 1
            for s in shape:
                size *= s
            return size

        elif self.type == "VALUE":
            # VALUE message are a value assignment for each var in the
            # separator of the sender
            return len(self.content[0]) * 2

    def __str__(self):
        return f"EfDpopMessage({self._msg_type}, {self._content})"


class AfDpopAlgo(VariableComputation):
    """
    AF-DPOP: Dynamic Programming Optimization Protocol

    When running this algorithm, the DFS tree must be already defined and the
    children, parents and pseudo-parents must be known.

    AF-DPOP computations support two kinds of messages:
    * UTIL message:
      sent from children to parent, contains a relation (as a
      multi-dimensional matrix) with one dimension for each variable in our
      separator.
    * VALUE messages :
      contains the value of the parent of the node and the values of all
      variables that were present in our UTIl message to our parent (that is
      to say, our separator) .
    """

    def __init__(self, comp_def: ComputationDef):
        # Initialize based on parent class
        super().__init__(comp_def.node.variable, comp_def)
        self._mode = comp_def.algo.mode

        # Check the arguments and parameters
        assert comp_def.algo.algo == "afdpop"
        assert type(self._variable.domain) is ContinuousDomain
        assert self._mode in ['min', 'max']
        # Currently variables with cost functions are not
        # supported. This functionality can be included 
        # by defining a unary relation for the variable
        assert type(self._variable) is Variable

        # Store the DFS relations
        self._parent, self._pseudo_parents, self._children, self._pseudo_children = get_dfs_relations(
            self.computation_def.node
        )
        descendants = self._pseudo_children + self._children
        self.logger.debug(f"Descendants for computation {self.name}: {descendants} ")

        self._children_separator = {}

        self._waited_children = []
        if not self.is_leaf:
            # If we are not a leaf, we must wait for the util messages from
            # our children.
            # This must be done in __init__ and not in on_start because we
            # may get an util message from one of our children before
            # running on_start, if this child computation start faster of
            # before us
            self._waited_children = list(self._children)

        # Filter the relations on all the nodes of the DFS tree to only keep the
        # relation on the on the lowest node in the tree that is involved in the
        # relation.
        self._constraints = []
        constraints = list(comp_def.node.constraints)
        for r in comp_def.node.constraints:
            # filter out all relations that depends on one of our descendants
            names = [v.name for v in r.dimensions]
            for descendant in descendants:
                if descendant in names:
                    constraints.remove(r)
                    break
        self._constraints = constraints
        self.logger.debug(
            f"Constraints for computation {self.name}: {self._constraints} "
        )

        # Algorithm specific
        # Number of discrete points to used per domain
        self._number_of_discrete_points = comp_def.algo.param_value("number_of_discrete_points")
        # Number of iterations for the update of the values
        self._number_of_iterations = comp_def.algo.param_value("number_of_iterations")
        # Threshold (for the absolute difference of the value step) used for termination 
        self._threshold = comp_def.algo.param_value("threshold")
        # Learning rate of the algorithm (during update of the values of the domains)
        self._alpha = comp_def.algo.param_value("alpha")

        # Internal representation of the utility message for the parent
        self._joined_utils = NAryMatrixRelation([], name=f"joined_utils_{self.name}")


    def footprint(self):
        return computation_memory(self.computation_def.node)

    @property
    def is_root(self) -> bool:
        return self._parent is None

    @property
    def is_leaf(self) -> bool:
        return len(self._children) == 0

    def on_start(self) -> None:
        """Called during startup of the agent"""
        if self.is_leaf and not self.is_root:
            # If the agent is a leaf in the pseudotree 
            # it can immediately compute the UTIL message and 
            # send it to its parent.
            # Note: as a leaf, the separator is the union of our parents 
            # and pseudo-parents
            util = self._compute_utils_msg()
            self.logger.info(
                f"Leaf {self._variable.name} init message {self._variable.name} -> {self._parent} : {util}"
            )
            msg = AfDpopMessage("UTIL", util)
            self.post_msg(self._parent, msg)

        elif self.is_leaf:
            # The agent is both root and leaf: 
            # means the agent is a isolated variable 
            # The agent can select its own value
            if self._constraints:

                # move the values
                self._move_values()

                # combined the constraints
                combined_constraints = CombinedNAryFunctionRelations(self._constraints)
                self._joined_utils = self._functional_join(
                    self._joined_utils, combined_constraints)

                # optimize the local variable
                current_cost, values = self._optimize_local_variable(
                    variable=self._variable, 
                    relation=self._joined_utils
                )

                # finish algorithm
                self.select_value_and_finish(values, float(current_cost))
            
            else:
                # If the variable is not constrained, we can simply take a value at
                # random:
                self.logger.debug(
                    f"Selecting random value for {self._variable.name} (not constrained)"
                )
                value = random() * \
                    (self._variable.domain.upper_bound - self._variable.domain.lower_bound) \
                    + self._variable.domain.lower_bound
                self.select_value_and_finish(value, 0.0)

    def stop_condition(self) -> None:
        # afdpop stop condition is easy at it only selects one single value!
        if self.current_value is not None:
            return ALGO_STOP
        else:
            return ALGO_CONTINUE

    def select_value_and_finish(self, value, cost: float) -> None:
        """Select a value for this variable.

        AF-DPOP is not iterative, once we have selected our value the algorithm
        is finished for this computation.

        Parameters
        ----------
        value: any (depends on the domain)
            the selected value
        cost: float
            the local cost for this value

        """

        self.value_selection(value, cost)
        self.stop()
        self.finished()
        self.logger.info(f"Value selected at {self.name} : {value} - {cost}")

    @register("UTIL")
    def _on_util_message(self, variable_name: str, recv_msg: AfDpopMessage, t: float) -> None:
        """
        Message handler for UTIL messages.

        Parameters
        ----------
        variable_name: str
            name of the variable that sent the message
        recv_msg: AfDpopMessage
            received message
        t: int
            message timestamp

        """
        self.logger.debug(f"UTIL from {variable_name} : {recv_msg.content} at {t}")
        utils = recv_msg.content

        # accumulate util messages until we got the UTIL from all our children
        self._joined_utils = self._functional_join(self._joined_utils, utils)
        try:
            self._waited_children.remove(variable_name)
        except ValueError as e:
            self.logger.error(
                f"Unexpected UTIL message from {variable_name} on {self.name} : {recv_msg} "
            )
            raise e
        # keep a reference of the separator of this children, we need it when
        # computing the value message
        self._children_separator[variable_name] = utils.dimensions

        if len(self._waited_children) == 0:
            if self.is_root:
                # We are the root of the DFS tree and have received all utils
                # we can select our own value and start the VALUE phase.

                # The root obviously has no parent nor pseudo parent, yet it
                # may have unary relations (with it-self!)
                combined_constraints = CombinedNAryFunctionRelations(self._constraints)
                self._joined_utils = self._functional_join(
                    self._joined_utils, 
                    combined_constraints)
                self._joined_utils = NAryMatrixRelation.from_func_relation(self._joined_utils)
                self._variable = self._joined_utils.dimensions[0]

                values, current_cost = find_arg_optimal(
                    self._variable, self._joined_utils, self._mode
                )
                selected_value = values[0]

                self.logger.info(
                    f"ROOT: On UTIL from {variable_name}, send VALUE to childrens {self._children} "
                )
                for c in self._children:
                    msg = AfDpopMessage("VALUE", ([self._variable], [selected_value]))
                    self.post_msg(c, msg)

                self.select_value_and_finish(selected_value, float(current_cost))
            else:
                # We have received the Utils msg from all our children, we can
                # now compute our own utils relation by joining the accumulated
                # util with the relations with our parent and pseudo_parents.
                util = self._compute_utils_msg()
                msg = AfDpopMessage("UTIL", util)
                self.logger.info(
                    f"On UTIL from {variable_name}, send UTIL to parent {self._parent} "
                )
                self.post_msg(self._parent, msg)

    def _update_constraint_variables(self, c: Constraint, variables: Iterable[Variable]) -> Constraint:
        """Update the variables of the constraint and return a new constraint"""
        # Create the list of updated constraints
        updated_constraint_variables = []
        variable_names = [v.name for v in variables]
        for variable_name in c.scope_names:
            # check if the variable is in the 'variables'
            if variable_name in variable_names:
                # add the updated version of the variable
                updated_constraint_variables.append(
                    variables[variable_names.index(variable_name)],
                )
            else:
                # keep the current variable
                updated_constraint_variables.append(
                    c.dimensions[c.scope_names.index(variable_name)],
                )

        # Construct the new constraint
        if isinstance(c, NAryFunctionRelation):
            # If the constraint is a functional relation, only the values of the domains
            # need to be updated, the utility values will calculated when required
            new_constraint = NAryFunctionRelation(
                f=c.function,
                variables=updated_constraint_variables,
                name=c.name,
            )
        elif isinstance(c, NAryMatrixRelation):
            # If the constraint is a matrix relation, additional utility values 
            # may need to be calculated (based on interpolation)

            # Create the new constraint (without utility values)
            new_constraint = NAryMatrixRelation(
                variables=updated_constraint_variables,
                matrix=None,
                name=c.name,
            )

            # Incrementally add the utility values
            for partial in generate_assignment_as_dict(updated_constraint_variables):
                # Check if the value is already in the constraint
                try:
                    new_val = c.get_value_for_assignment(
                        var_values=partial
                    )
                except ValueError as e:
                    # The value was not in the 'original' utility matrix
                    # therefore it will be interpolated
                    # print(f'ValueError: {e}')
                    new_val = np.nan

                # add the new (utility) value for the partial assignment
                new_constraint = new_constraint.set_value_for_assignment(
                    partial, new_val)

            # Interpolate all the value (indicated by nan)
            nans, x = self.nan_helper(new_constraint._m)
            new_constraint._m[nans]= np.interp(
                x(nans), x(~nans), new_constraint._m[~nans])

        return new_constraint

    @staticmethod
    def nan_helper(y):
        """Helper to handle indices and logical indices of NaNs.

        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
            to convert logical indices of NaNs to 'equivalent' indices
        Example:
            >>> # linear interpolation of NaNs
            >>> nans, x= nan_helper(y)
            >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])

        Source:
            https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
        """

        return np.isnan(y), lambda z: z.nonzero()[0]

    def _compute_utils_msg(self) -> NAryMatrixRelation:
        """Returns an optimal matrix relation based on projection of own variable.

        Compute the utility message by moving the values of the variables in the
        separator of the agent and then calculating the optimized utility for the
        new values with respect to the optimal local variable(s).
        """
        # Move all the domain values within the separator of the agent
        self._move_values()

        # Join all the constraints
        combined_constraints = CombinedNAryFunctionRelations(self._constraints)
        self._joined_utils = self._functional_join(
            self._joined_utils, 
            combined_constraints)

        # use projection to eliminate self out of the message to our parent
        util = self._functional_projection(
            combined_constraints, 
            self._variable)

        return util

    def _functional_projection(self, a_rel: Constraint, a_var: Variable) -> NAryMatrixRelation:
        """Returns the projection of the constraint as a matrix relation

        The projection of a relation `a_rel` along the variable `a_var` is the
        optimization of the matrix along the axis of this variable.

        The result of `projection(a_rel, a_var)` is also a relation, with one less
        dimension than a_rel (the a_var dimension).
        For each possible instantiation of the variable other than a_var,
        the optimal instantiation for a_var is chosen and the corresponding
        utility recorded in projection(a_rel, a_var)

        Also see definition in Petcu 2007.

        Parameters
        ----------
        a_rel: Constraint
            the projected relation
        a_var: Variable
            the variable over which to project
        mode: mode as str
            'max (default) for maximization, 'min' for minimization.

        Returns
        -------
        Constraint:
            the new relation resulting from the projection
        """
        # Collect the variables that remain after projection
        remaining_vars = a_rel.dimensions.copy()
        remaining_vars.remove(a_var)

        # Create the new relation resulting from the projection
        proj_rel = NAryMatrixRelation(remaining_vars)

        # Fill the values of the new relation
        for partial in generate_assignment_as_dict(remaining_vars):
            optimal_utility, _ = self._optimize_local_variable(a_rel.slice(partial), a_var)
            proj_rel = proj_rel.set_value_for_assignment(partial, optimal_utility)

        return proj_rel
    
    def _optimize_local_variable(self, relation: Constraint, variable: Variable) -> [float, float]:
        """Return the optimal utility and value for the relation
        
        :param relation: a function or an object implementing the Relation
        protocol and depending only on the var 'variable'
        :param variable: the variable

        :return: the optimal utility and the value (argmax).
        """
        # Check the relation (only dependent on the local variable)
        if hasattr(relation, "dimensions"):
            if relation.arity != 1 or relation.dimensions[0] != variable:
                raise ValueError(
                    "For _optimize_local_variable, the relation must depend "
                    "only on the given variable : {} {}".format(relation, variable)
                )

        # Adjust the factor for the optimization mode
        if self._mode == 'min':
            factor = 1.0
        elif self._mode == 'max':
            factor = -1.0

        # Execute the (scalar) bounded optimization
        opt_res = optimize.minimize_scalar(
            fun=lambda x: factor * relation(x),
            bounds=(
                variable.domain.lower_bound, 
                variable.domain.upper_bound,
                ),
            method='bounded')

        # Post-process the results
        optimal_utility = factor * opt_res.fun
        optimal_value = opt_res.x

        return optimal_utility, optimal_value

    def _functional_join(self, u1: Constraint, u2: Constraint) -> Constraint:
        """Returns a new Constraint by joining the two Constraints u1 and u2.

        Adjusted join functionality that holds the different domains of the variables
        into account.
        The different domains are a feature of the AF-DPOP algorithm in which the
        values of the domains 'move' during the UTIL-phase.

        The dimension of the new Constraint is the union of the dimensions of u1
        and u2. For any complete assignment, the value of this new relation is the sum of
        the values from u1 and u2 for the subset of this assignment that apply to
        their respective dimension.

        For more details, see the definition of the join operator in Petcu's Phd thesis.

        Dimension order is important for some operations, variables for u1 are
        listed first, followed by variables from u2 that where already used by u1
        (in the order in which they appear in u2.dimension).
        Note that relying on dimension order is fragile and discouraged,
        use keyword arguments whenever possible instead!

        Parameters
        ----------
        u1: Constraint
            n-ary relation
        u2: Constraint
            n-ary relation

        Returns
        -------
        Constraint:
            a new Constraint
        """
        # Since the domains of the variables can be different, this needs to be checked first
        update_variables = []
        u2_variable_names = [v.name for v in u2.dimensions]
        for d in u1.dimensions:
            if d.name in u2_variable_names:
                update_variables.append((d, u2.dimensions[u2_variable_names.index(d.name)]))

        # Update the constraints by extending the domains of the variables 
        for update_variable_pair in update_variables:
            # print(f'Updating variable pair: {update_variable_pair}')
            # print('')
            # Get the domains
            domains = [v.domain.values for v in update_variable_pair]

            # Check if they are equal
            domains_equal = domains[0] == domains[1]

            if not domains_equal:
                # Create a unified domain
                unified_domain = []
                unified_domain.extend(domains[0])
                unified_domain.extend(domains[1])
                unified_domain = set(unified_domain)

                # Create a new variable
                updated_variable = self._update_variable_values(
                    variable=update_variable_pair[0], 
                    values=unified_domain)

                # If this variable is the local variable, update it accordingly
                if updated_variable.name == self._variable.name:
                    self._variable = updated_variable

                # Create a new constraint
                u1 = self._update_constraint_variables(u1, [updated_variable])
                u2 = self._update_constraint_variables(u2, [updated_variable])

        # Now the constraint have equal variable (domains)
        # they can be joined (as before)
        if isinstance(u1, NAryMatrixRelation) and isinstance(u2, NAryMatrixRelation):
            # use the conventional join
            joined_relation = join(u1, u2)
        elif isinstance(u1, NAryFunctionRelation) and isinstance(u2, NAryFunctionRelation):
            # both are functional
            joined_relation = CombinedNAryFunctionRelations([u1, u2])
        elif isinstance(u1, NAryFunctionRelation):
            # u1 function, u2 matrix
            joined_relation = self._join_matrix_function(matrix_relation=u2, function_relation=u1)
        else:
            # u2 function, u1 matrix
            joined_relation = self._join_matrix_function(matrix_relation=u1, function_relation=u2)

        return joined_relation

    def _join_matrix_function(self, matrix_relation: NAryMatrixRelation, function_relation: NAryFunctionRelation) -> "CombinedNAryFunctionRelations":
        """Returns a new relation from the combined matrix and function relation"""
        # Check the arguments
        if matrix_relation.arity == 0:
            # empty matrix relations
            # return the function relations
            return function_relation
        elif function_relation.arity == 0:
            # empty matrix relation
            # return a functional relation
            return InterpolatedMatrixRelation(matrix_relation)

        # Join both (functional) relations
        joined_relation  = CombinedNAryFunctionRelations([
            InterpolatedMatrixRelation(matrix_relation), 
            function_relation])

        return joined_relation 

    def _update_variable_values(self, variable: Variable, values: Iterable) -> Variable:
        """Returns a copy of the variable with discrete domain
        
        Parameters
        ----------
        variable: Variable
            variable instance to be updated
        values: Iterable
            an array containing the values allowed for the
            variables with this domain.
        
        Returns
        -------
        New variable instance with updated domain
        """
        values = sorted(values)
        discrete_domain = Domain(
            variable.domain.name, 
            variable.domain.type, 
            values,
        )
        new_variable = Variable(
            variable.name, 
            discrete_domain, 
            variable.initial_value,
        )
        return new_variable

    def _move_values_of_variable(self, variable: Variable) -> Variable:
        """Move the values of the domain, returns as a new variable
        
        Based on Equation 17:
        v_{i_j} = v_{i_j} + alpha frac{partial f_{i_j}(x_i, x_{i_j}) }{ partial x_{i_j} }
            |^{v_{i_j}}_{argmax_{x_i} f_{i_j} ( x_{i_j} = v_{i_j} ) }
        """
        # Do not move own variable
        # (this is done during optimization)
        if variable.name == self._variable.name:
            return variable

        # Create the initial values of the domain
        # by selecting discrete points
        initial_values = np.linspace(
            start=variable.domain.lower_bound, 
            stop=variable.domain.upper_bound, 
            num=self._number_of_discrete_points, 
            endpoint=True,
        )
        # Loop over all the values within the domain
        updated_values = []
        for domain_value in initial_values:
            # Start optimization based on local gradient
            found_optimum = False
            # Iteratively update the value of the domain
            for iteration_index in range(self._number_of_iterations):
                # Get all the constraints that are relevant for the variable
                # f_{i_j}
                relevant_constraints = [c for c in self._constraints if variable.name in c.scope_names]
                combined_constraints = CombinedNAryFunctionRelations(relevant_constraints)

                # Set the partial assignment for all variables
                # f_{i_j} ( x_{i_j} = v_{i_j} )
                relevant_constraints_sliced = [c.slice({variable.name: domain_value}) for c in relevant_constraints]

                # Combine all the constraints in a single constraint
                combined_constraints_sliced = CombinedNAryFunctionRelations(relevant_constraints_sliced)

                # calculate the optimal value for the local variable
                # argmax_{x_i} f_{i_j} ( x_{i_j} = v_{i_j} ) 
                _, value_argmax = self._optimize_local_variable(
                    combined_constraints_sliced, 
                    self._variable, 
                )

                if found_optimum:
                    # print(f'The variable {variable.name} could be optimized directly,
                    # therefore the optimum is found @ {value_argmax}')
                    updated_value = value_argmax
                    break
        
                # UPDATE: since the derivative cannot be assumed to be available,
                # approximate the (partial) derivative
                dxij = 1e-3
                y1 = combined_constraints.get_value_for_assignment({
                    variable.name: domain_value + dxij,
                    self._variable.name: value_argmax
                })
                y2 = combined_constraints.get_value_for_assignment({
                    variable.name: domain_value - dxij,
                    self._variable.name: value_argmax
                })
                df_approx = (y1 - y2) / (2. * dxij)
                value_delta = self._alpha * df_approx

                # Calculate the updated domain value
                if self._mode == 'min':
                    updated_value = domain_value - value_delta
                elif self._mode == 'max':
                    updated_value = domain_value + value_delta

                # Bound the new value by the bounds
                # of the domain
                updated_value = np.min(
                    [np.max([updated_value, 
                             variable.domain.lower_bound]), 
                             variable.domain.upper_bound],
                )

                # Check the threshold for convergence
                if np.abs(value_delta) <= self._threshold:
                    break

                # Update the domain value and restart the loop
                domain_value = updated_value

            # Add the new/updated value to the list of 
            # new domain values
            updated_values.append(updated_value)

        # Create a new variable based on the new values
        new_variable = self._update_variable_values(
            variable=variable, 
            values=updated_values,
        )
        return new_variable

    def _move_values(self) -> None:
        """Moves the values of the domains within its seperator based on Equation 17"""
        # Get a list of all variables that are in the separator of the agent
        # based on the constraints
        separator_variables = list(set([d for c in self._constraints for d in c.dimensions]))

        # Move the values of the variables
        updated_variables = []
        for variable in separator_variables:
            updated_variables.append(self._move_values_of_variable(variable))

        # Update the values of the variables for all the constraints
        new_constraints = []
        for c in self._constraints:
            new_constraint = self._update_constraint_variables(c, updated_variables)
            new_constraints.append(new_constraint)
        self._constraints = new_constraints

    @register("VALUE")
    def _on_value_message(self, variable_name: str, recv_msg: AfDpopMessage, t: float) -> None:
        """
        Message handler for VALUE messages.

        Parameters
        ----------
        variable_name: str
            name of the variable that sent the message
        recv_msg: AfDpopMessage
            received message
        t: int
            message timestamp
        """
        self.logger.debug(
            f"{self.name}: on value message from {variable_name} : '{recv_msg}' at {t}"
        )

        value = recv_msg.content

        # Value msg contains the optimal assignment for all variables in our
        # separator : sep_vars, sep_values = value
        value_dict = {k.name: v for k, v in zip(*value)}
        self.logger.debug(f"Slicing relation on {value_dict}")

        # as the value msg contains values for all variables in our
        # separator, slicing the util on these variables produces a relation
        # with a single dimension, our own variable.
        rel = self._joined_utils.slice(value_dict)

        self.logger.debug(f"Relation after slicing {rel}")

        optimal_utility, selected_value = self._optimize_local_variable(
            relation=rel,
            variable=self._variable,
            )

        for c in self._children:
            variables_msg = [self._variable]
            values_msg = [selected_value]

            # own_separator intersection child_separator union
            # self.current_value
            for v in self._children_separator[c]:
                try:
                    values_msg.append(value_dict[v.name])
                    variables_msg.append(v)
                except KeyError:
                    # we want an intersection, we can ignore the variable if
                    # not in value_dict
                    pass
            msg = AfDpopMessage("VALUE", (variables_msg, values_msg))
            self.post_msg(c, msg)

        self.select_value_and_finish(selected_value, float(optimal_utility))


class InterpolatedMatrixRelation(AbstractBaseRelation):
    """
    A functional relation based on a matrix relation.
    The values from the matrix relation are interpolated
    when the functional relation is called
    """
    def __init__(self, matrix_relation: NAryMatrixRelation) -> None:
        """Convert the matrix relation into a functional relation"""
        # Initialize the relation
        super().__init__(matrix_relation.name)

        # Create the interpolator for the matrix values
        self._create_interpolator(matrix_relation)
        self._dimensions = matrix_relation.dimensions
        self._variables = self._dimensions
        self._var_mapping = {v.name: v.name for v in self._dimensions}

    def slice(self, partial_assignment: Dict[str, object]) -> RelationProtocol:
        raise NotImplementedError(
            "slice is not implemented for interpolated relations"
        )

    def set_value_for_assignment(self, assignment: Dict[str, object], relation_value) -> None:
        raise NotImplementedError(
            "set_value_for_assignment is not "
            "implemented for function-defined relations"
        )

    def get_value_for_assignment(self, assignment: Dict[str, object]) -> float:

        if isinstance(assignment, list):
            args_dict = {}
            for i in range(len(assignment)):
                arg_name = self._var_mapping[self._variables[i].name]
                args_dict[arg_name] = assignment[i]
            return self._get_interpolated_value(**args_dict)

        elif isinstance(assignment, dict):
            args_dict = {}
            for var_name in assignment:
                arg_name = self._var_mapping[var_name]
                args_dict[arg_name] = assignment[var_name]
            return self._get_interpolated_value(**args_dict)

        else:
            raise ValueError("Assignment must be list or dict")

    def __call__(self, *args, **kwargs):
        if not kwargs:
            if len(args) == 1 and type(args[0]) is dict:
                return self(**args[0])
            return self.get_value_for_assignment(list(args))
        else:
            return self.get_value_for_assignment(kwargs)

    def __repr__(self):
        return "InterpolatedMatrixRelation({}, {})".format(self.name, self._variables)

    def __str__(self):
        return "InterpolatedMatrixRelation({})".format(self._name)

    def __eq__(self, other):
        if type(other) != InterpolatedMatrixRelation:
            return False
        if (
            self.name == other.name
            and other.dimensions == self.dimensions
            and self._f == other.function
        ):
            return True
        return False

    def __hash__(self):
        return hash((self.name, tuple(self._variables), self._f))

    def _create_interpolator(self, matrix_relation: NAryMatrixRelation) -> None:
        """Create the interpolator for the matrix values"""
        if matrix_relation.arity == 1:
            points = []
            values = []
            for assignment in generate_assignment_as_dict(matrix_relation.dimensions):
                points.append([v for x, v in assignment.items()])
                values.append(matrix_relation.get_value_for_assignment(assignment))
            points_array = np.atleast_2d(points).flatten()
            values_array = np.array(values)
            self._f = interp1d(points_array, values_array, kind="linear")
        else:
            domains = []
            for v in matrix_relation.dimensions:
                domains.append(np.array(v.domain.values))
            points_mesh = np.meshgrid(*domains, indexing='ij', sparse=False)

            points_shape = points_mesh[0].shape
            number_of_points = np.prod(points_shape)

            values_array = np.empty([number_of_points, 1])
            points_array = np.empty([number_of_points, len(matrix_relation.dimensions)])
            for i in range(number_of_points):
                index = np.unravel_index(i, points_shape)
                points_array[i, :] = [p[index] for p in points_mesh]
                values_array[i] = matrix_relation(*points_array[i, :])
            self._f = LinearNDInterpolator(points_array, values_array)
           
    def _get_interpolated_value(self, **kwargs) -> float:
        """Return the interpolated value"""
        x = [kwargs[v_name] for v_name in self._var_mapping.keys()]
        value = float(self._f(x).flatten())
        if np.isnan(value):
            return -np.inf
        else:
            return value


class CombinedNAryFunctionRelations(AbstractBaseRelation):
    """
    A list of NAryFunctionRelation used to combine
    multiple relations into a single relation.
    """
    def __init__(
        self,
        relations: Iterable[NAryFunctionRelation],
        name: str=None,
    ) -> None:
        """
        Combined NAryFunctions into a single relation.

        :param relations: the list of NAryFunctionRelation objects
        this relation depends on.
        """
        super().__init__(name)

        # Store the relations
        self._relations = relations
        # get all the dimensions
        relation_dimensions = [v 
            for r in self._relations 
            for v in r.dimensions]
        # store a unique list
        self._variables = list(dict.fromkeys(relation_dimensions))
        self._dimensions = self._variables


    def __call__(self, *args, **kwargs) -> float:
        if not kwargs:
            if len(args) == 1 and type(args[0]) is dict:
                return self(**args[0])
            return self.get_value_for_assignment(list(args))
        else:
            return self.get_value_for_assignment(kwargs)

    def slice(self, partial_assignment: Dict[str, object]) -> object:
        """Return a CombinedNAryFunctionRelations object based on the slice"""
        # Loop over the relations
        sliced_relations = []
        for relation in self._relations:
            # Pre-process the partial_assignment
            relation_partial_assignment = {v.name: partial_assignment[v.name] 
                for v in relation._variables 
                if v.name in partial_assignment}
            # Slice the relation
            sliced_relation = relation.slice(relation_partial_assignment)
            # Add to the list
            sliced_relations.append(sliced_relation)
        # Create a new CombinedNAryFunctionRelations object
        return CombinedNAryFunctionRelations(sliced_relations)

    def get_value_for_assignment(self, assignment: Dict[str, object]) -> float:
        """Return the combined (by summation) value for the assignment"""
        # Loop over the relations
        values = []
        for relation in self._relations:
            # Pre-process the assignment
            if isinstance(assignment, dict):
                relation_assignment = {v.name: assignment[v.name] for v in relation._variables if v.name in assignment}
            else:
                relation_assignment = assignment
            # Get the value for the assignment
            relation_value = relation.get_value_for_assignment(relation_assignment)
            values.append(relation_value)
        # Combined the values by summation
        return np.sum(values)
