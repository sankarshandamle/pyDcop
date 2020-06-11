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

DBAY: Distributed Bayesian algorithm
-----------------------------------------------

Distributed Bayesian: a continuous Distributed Constraint Optimization Problem solver
Fransman, J., Sijs, J., Dol, H., Theunissen, E., & De Schutter, B. (2020).
https://arxiv.org/abs/2002.03252

Algorithm Parameters
^^^^^^^^^^^^^^^^^^^^

* **number_of_samples**: the amount of samples that is used as termination criterion.
  Defaults to 10

* **acquisition_optimization**: # the method for the acquisition function optimization.
  Defaults to 'brute'

Example
^^^^^^^
::

    pydcop solve --algo dbay tests/instances/cdcop_bird_func.yaml

"""
from random import choice
from typing import Iterable, Optional, Dict, Any

import copy
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize_scalar, shgo, differential_evolution, dual_annealing, brute 

from pydcop.computations_graph.pseudotree import get_dfs_relations
from pydcop.infrastructure.computations import Message, VariableComputation, register
from pydcop.dcop.objects import Variable, ContinuousDomain
from pydcop.dcop.relations import (
    NAryMatrixRelation,
    Constraint,
    find_arg_optimal,
    join,
    projection,
)
from pydcop.algorithms import (
    ALGO_STOP,
    ALGO_CONTINUE,
    ComputationDef,
    AlgoParameterDef,
)

GRAPH_TYPE = "pseudotree"

algo_params = [
    # Number of samples used as threshold during optimization of the local variable
    AlgoParameterDef("number_of_samples", "int", None, 10),
    # Select the method for the acquisition function optimization
    AlgoParameterDef("acquisition_optimization", "str", None, 'brute'),
]

def build_computation(comp_def: ComputationDef):
    computation = DBayAlgo(comp_def)
    return computation

def computation_memory(*args):
    raise NotImplementedError("DBay has no computation memory implementation (yet)")

def communication_load(*args):
    raise NotImplementedError("DBay has no communication_load implementation (yet)")


class DBayMessage(Message):
    def __init__(self, msg_type, content):
        super(DBayMessage, self).__init__(msg_type, content)

    @property
    def size(self):
        # DBay messages
        # UTIL : single value for every value message from parent
        # VALUE : partial assignment from parent
        # FINAL : final partial assignment from parent
        # LIPSCHITZ : combined Lipschitz constants

        if self.type == "UTIL":
            # single value for every value message from parent
            return 1

        elif self.type == "VALUE":
            # VALUE message are a value assignment for each var in the
            # separator of the sender
            return len(self.content) * 2

        elif self.type == "FINAL":
            # VALUE message are a value assignment for each var in the
            # separator of the sender
            return len(self.content) * 2

        elif self.type == "LIPSCHITZ":
            # Combined Lipschitz constants of all agents within the
            # separator of the sender
            return 1

    def __str__(self):
        return f"DBayMessage({self._msg_type}, {self._content})"


class DBayAlgo(VariableComputation):
    """
    DBAY: Distributed Bayesian algorithm
    -----------------------------------------------

    Distributed Bayesian: a continuous Distributed Constraint Optimization Problem solver
    Fransman, J., Sijs, J., Dol, H., Theunissen, E., & De Schutter, B. (2020).
    https://arxiv.org/abs/2002.03252

    """

    def __init__(self, comp_def: ComputationDef) -> None:
        # Initialize based on parent class
        super().__init__(comp_def.node.variable, comp_def)
        self._mode = comp_def.algo.mode

        # Check the arguments and parameters
        assert comp_def.algo.algo == "dbay"
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
            # If we are not a leaf, we must wait for the LIPSCHITZ messages
            # from our children.
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
        # Check if all constraints have the correct properties
        for c in self._constraints:
            assert not (c.get_property('Lipschitz_constant') is None)

        self.logger.debug(
            f"Constraints for computation {self.name}: {self._constraints} "
        )

        # Algorithm specific
        # Parameters
        # Number of samples used as threshold during optimization of the local variable
        self._max_number_of_samples = comp_def.algo.param_value("number_of_samples")
        # Select the method for the acquisition function optimization
        self._acquisition_optimization = comp_def.algo.param_value("acquisition_optimization")

        # Problem specific
        self._Lipschitz_constant = 0.0

        # Internal buffers
        self._stored_results = []
        self._child_utility = []
        self._current_sample = {}
        self._finishing = False

    def footprint(self):
        return computation_memory(self.computation_def.node)

    @property
    def is_root(self) -> bool:
        return self._parent is None

    @property
    def is_leaf(self) -> bool:
        return len(self._children) == 0

    def on_start(self) -> None:
        """Execute initial processes, called during start of algorithm

        Start combinging the Lipschitz constants from the constraints.
        The process starts from the leaves.
        """
        if self.is_leaf:
            # Combined the Lipschitz constants from all constraints
            self._Lipschitz_constant = self._combined_Lipschitz_constants()
            # Send the message to the parent
            self._send_message_to_parent(
                content=self._Lipschitz_constant,
                message_type='LIPSCHITZ',
            )

    def _combined_Lipschitz_constants(self) -> float:
        """Returns the combined Lipschitz constants for all constraints"""
        Lipschitz_constant = 0.0
        for c in self._constraints:
            Lipschitz_constant += c.get_property('Lipschitz_constant')
        return Lipschitz_constant

    def select_value_and_finish(self, value: float, cost: float) -> None:
        """Select a value for this variable.

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

    def _send_message_to_all_children(self, content: Any, message_type: str) -> None:
        """Send a message to all children based on message type"""
        # Create the message
        msg = DBayMessage(message_type, content)
        # Send to all children
        for child in self._children:
            self.post_msg(target=child, msg=msg)

    def _send_message_to_parent(self, content: Any, message_type: str) -> None:
        """Send a message to the parent based on the message type"""
        # Create the message
        msg = DBayMessage(message_type, content)
        # Send to parent
        self.post_msg(target=self._parent, msg=msg)

    @register('LIPSCHITZ')
    def _on_lipschitz_message(self, variable_name: str, recv_msg: DBayMessage, t: float) -> None:
        """Message handler for LIPSCHITZ messages.

        Parameters
        ----------
        variable_name: str
            name of the variable that sent the message
        recv_msg: DBayMessage
            received message
        t: int
            message timestamp

        """
        self.logger.debug(f"LIPSCHITZ from {variable_name} : {recv_msg.content} at {t}")

        # accumulate messages until we got the LIPSCHITZ messages from all our children
        try:
            self._waited_children.remove(variable_name)
        except ValueError as e:
            self.logger.error(
                f"Unexpected LIPSCHITZ message from {variable_name} on {self.name} : {recv_msg} "
            )
            raise e

        # process the Lipschitz constraint
        Lipschitz_constant = recv_msg.content
        self._Lipschitz_constant += Lipschitz_constant

        # Check if all messages have been received
        if len(self._waited_children) == 0:
            if self.is_root:
                # The root starts with sending the initial VALUE message
                # All other agents wait to receive a VALUE message
                self._optimize_for_sample(None)
            else:
                # Add the combined Lipschitz constant from all constraints
                self._Lipschitz_constant += self._combined_Lipschitz_constants()
                # Send the message to the parent
                self._send_message_to_parent(
                    content=self._Lipschitz_constant,
                    message_type='LIPSCHITZ',
                )

    @register('UTIL')
    def _on_util_message(self, variable_name: str, recv_msg: DBayMessage, t: float) -> None:
        """Message handler for UTIL messages.

        Parameters
        ----------
        variable_name: str
            name of the variable that sent the message
        recv_msg: DBayMessage
            received message
        t: int
            message timestamp

        """
        self.logger.debug(f"UTIL from {variable_name} : {recv_msg.content} at {t}")

        # accumulate util messages until we got the UTIL from all our children
        try:
            self._waited_children.remove(variable_name)
        except ValueError as e:
            self.logger.error(
                f"Unexpected UTIL message from {variable_name} on {self.name} : {recv_msg} "
            )
            raise e

        # process the utility
        child_utility = recv_msg.content
        self._child_utility.append(child_utility)

        # Check if all messages have been received
        if len(self._waited_children) == 0:
            # continue processing the utility messages
            self._process_utility_for_sample()

    @register('VALUE')
    def _on_value_message(self, variable_name: str, recv_msg: DBayMessage, t: float) -> None:
        """Message handler for VALUE messages.

        Parameters
        ----------
        variable_name: str
            name of the variable that sent the message
        recv_msg: DBayMessage
            received message
        t: int
            message timestamp
        """
        self.logger.debug(
            f"VALUE from {variable_name} : {recv_msg.content} at {t}"
        )

        # Get the sample (from the parent)
        parent_sample = recv_msg.content

        # Process the samples
        self._optimize_for_sample(parent_sample)

    def _optimize_for_sample(self, sample: Dict[str, float]) -> None:
        """Process the sample from the parent and optimize variables accordingly"""
        # Create a (new) instance of the bayesian optimizer
        self._stored_results = []
        self._bay = BayesianOptimizer(
            Lipschitz_constant=self._Lipschitz_constant,
            lower_bound=self._variable.domain.lower_bound,
            upper_bound=self._variable.domain.upper_bound,
            initial_samples=[
                self._variable.domain.lower_bound,
                self._variable.domain.upper_bound,
            ],
            acquisition_optimization=self._acquisition_optimization,
        )

        # Optimize the local variables based on the parent sample
        self._optimize_local_variable(sample)

    def _optimize_local_variable(self, sample: Dict[str, float]) -> None:
        """Optimize the local variables through Bayesian sampling"""
        # Compute the (optimal) next sample through Bayesian optimization
        local_sample = self._compute_optimal_sample()

        # Augment the sample with the local sample
        if sample:
            self._current_sample = dict(sample, **local_sample)
        else:
            self._current_sample = copy.deepcopy(local_sample)

        # If the agent is a leaf node, it can process the utility
        # directly without 'consulting' other agents.
        if self.is_leaf:
            self._process_utility_for_sample()
        else:
            # Create the buffer for the utility messages from the children
            self._child_utility = []

            # Store the names of the children that need to return a message
            self._waited_children = list(self._children)

            # Send the value message to all children
            self._send_message_to_all_children(
                content=self._current_sample,
                message_type='VALUE',
            )

    def _compute_optimal_sample(self) -> Dict[str, float]:
        """Return the optimal sample based on Bayesian optimization"""
        # Compute the optimal sample
        sample_value = self._bay.compute_next_sample()
        # Construct a sample dict object
        sample = {self._variable.name: sample_value}
        return sample

    def _calculate_local_utility(self) -> float:
        """Return the combined local utility for the current sample"""
        # Initiate the utility value
        utility_value = 0.0

        # Loop over all constraints
        for c in self._constraints:
            utility_value += c.get_value_for_assignment({
                variable_name: self._current_sample[variable_name]
                for variable_name in c.scope_names})

        # Adjust the factor for the optimization mode
        if self._mode == 'min':
            factor = -1.0
        elif self._mode == 'max':
            factor = 1.0

        return factor * utility_value

    def _process_utility_for_sample(self) -> None:
        """Process the received utility messages"""
        # Calculate the local utility
        self._local_utility = self._calculate_local_utility()

        # Append the child utility to the local utility
        utility_value = self._local_utility + np.sum(self._child_utility)

        # Add the sample/utility as observation to the Bayesian optimizer
        success = self._bay.add_sample(
            self._current_sample[self._variable.name],
            utility_value,
        )

        if success:
            # Store the utility of the sample locally
            self._stored_results.append(
                [copy.deepcopy(self._current_sample), utility_value])
            # Check the threshold
            threshold_reached = self._check_threshold()
        else:
            # Adding samples failed, 
            # probably because the sample has already been sampled
            threshold_reached = True

        # Continue with the sample phase
        if threshold_reached:
            # After optimization, the root starts final phase
            # other agents send the utility message to their parents
            if self.is_root or self._finishing:
                self._process_final()
            else:
                # Get the optimal local sample based on the sample message of the parent
                [_, optimal_utility] = self._retrieve_optimal_sample_from_stored()

                # Send optimal message to parent
                self._send_message_to_parent(
                    content=optimal_utility,
                    message_type='UTIL')
        else:
            # Continue with optimization of the local variable
            self._optimize_local_variable(self._current_sample)

    def _check_threshold(self) -> bool:
        """Return True if the threshold has been reached"""
        # Check based on the number of samples
        self.logger.debug(
            f'Threshold: {self._bay.number_of_samples} >= {self._max_number_of_samples}'
        )
        threshold_reached = self._bay.number_of_samples >= self._max_number_of_samples

        return threshold_reached

    # FINAL RELATED
    # VALUE MESSAGE RELATED
    @register('FINAL')
    def _on_final_message(self, variable_name: str, recv_msg: DBayMessage, t: float) -> None:
        """Message handler for FINAL messages.

        Parameters
        ----------
        variable_name: str
            name of the variable that sent the message
        recv_msg: DBayMessage
            received message
        t: int
            message timestamp
        """
        self.logger.debug(
            f"{self.name}: on final message from {variable_name} : '{recv_msg}' at {t}"
        )

        # Get the sample (from the parent)
        final_sample = recv_msg.content

        # Update the finishing flag
        self._finishing = True

        if self.is_root:
            # Send the sample
            self._process_final()
        else:
            # Optimize based on the final sample
            self._optimize_for_sample(final_sample)

    def _process_final(self) -> None:
        """Finalize the algorithm by sending the final message to all children and selecting the optimal value"""
        # Get the optimal local sample based on the final message of the parent
        optimal_local_sample, optimal_utility = self._retrieve_optimal_sample_from_stored()

        # Send the optimal_sample to all children
        self._send_message_to_all_children(
            content=optimal_local_sample,
            message_type='FINAL')

        # Select the optimal value and finish the algorithm
        optimal_local_value = optimal_local_sample[self._variable.name]
        self.select_value_and_finish(optimal_local_value, optimal_utility)

    def _retrieve_optimal_sample_from_stored(self) -> [Dict[str, float], float]:
        """Returns the optimal sample and the corresponding utility"""
        # Get the optimal utility value
        max_utility = np.max([u for [s, u] in self._stored_results])
        # Get the optimal sample
        optimal_sample = [s for [s, u] in self._stored_results if u == max_utility][0]

        return [optimal_sample, max_utility]


class BayesianOptimizer(object):
    """General class for Bayesian optimization.

    Based on samples of an objective function,
    the regression calculates the 'mean' and 'covariance'.
    These values are used by the acquisition function to select new sample.
    """
    def __init__(self,
                 Lipschitz_constant: float=1.0,
                 lower_bound: float=0.0,
                 upper_bound: float=1.0,
                 initial_samples=[],
                 acquisition_optimization='brute',
                 ) -> None:
        # Store the arguments
        self._Lipschitz_constant = Lipschitz_constant
        self._initial_samples = initial_samples
        self._acquisition_optimization_brute = (acquisition_optimization == 'brute')

        # Store the bounds of the domain for normalization
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

        # Define the acquisition function
        self._acquisition = AcquisitionExpected(xi=0.0)

        # Construct the regressor
        Lipschitz_constant_normalized = self._Lipschitz_constant * \
            np.abs(self._upper_bound - self._lower_bound)
        self._gp = MarkovianRegressor(Lipschitz_constant_normalized)

    @property
    def number_of_samples(self) -> int:
        """Return the number of samples"""
        return self._gp.number_of_samples

    def add_sample(self, sample: float, value: float) -> bool:
        """Add the sample to the internal model"""
        # Normalize the sample
        sample_X_normalized = self._normalize_sample(sample)

        # Update the regressor
        success = self._gp.add_sample(sample_X_normalized, value)

        # Update the maximum value property of the acquisition function
        max_value = self._gp.get_maximum_value()
        self._acquisition.update_max_value(max_value)

        return success

    def compute_next_sample(self) -> float:
        """Return the next sample based on the acquisition function"""
        # Check if the initial samples have been sampled
        if len(self._initial_samples) > 0:
            # select the first sample
            next_sample = self._initial_samples.pop(0)
        else:
            # Optimize the acquisition function to select the new value
            next_sample_normalized = self._optimize_acquisition()
            next_sample = self._denormalize_sample(next_sample_normalized)

        return next_sample

    def _get_acquisition_value(self, x: float) -> float:
        """Return the value of the acquisition function for sample 'x'"""
        # update the mean and standard deviation evaluated at 'x'
        [mu, sigma] = self._compute_posterior(x)

        # Calculate the acquisition function
        acquisition_value = self._acquisition.compute(mu=mu, sigma=sigma)

        return acquisition_value

    def _optimize_acquisition(self) -> float:
        """Return the optimal sample value based on the acquisition function"""
        if self._acquisition_optimization_brute:
            # Apply a brute force method to optimize the 
            # value for the acquisition function
            x = np.linspace(0.0, 1.0, num=1000)
            y = self._get_acquisition_value(x)
            optimal_sample = x[np.argmax(y)]
        else:
            # Apply a negative factor as dual_annealing 
            # is a global minimization algorithm
            opt_res = dual_annealing(
                lambda x: -1.0 * self._get_acquisition_value(x),
                bounds=[(0.0, 1.0)],
                maxiter=10,
            )
            optimal_sample = float(opt_res.x)

        return optimal_sample

    def _compute_posterior(self, x: float) -> [Optional[float], Optional[float]]:
        """Return the mean and standard deviation based on the samples x"""
        [mean, standard_deviation] = self._gp.predict(x)
        if mean is None or standard_deviation is None:
            return [None, None]
        return [mean, standard_deviation]

    def _normalize_sample(self, x: float) -> float:
        """Return the normalized value of the sample x based on interval bounds"""
        return (x - self._lower_bound) / (self._upper_bound - self._lower_bound)

    def _denormalize_sample(self, normalized_x: float) -> float:
        """Return the denormalized value of the sample x based on interval bounds"""
        return normalized_x * (self._upper_bound - self._lower_bound) + self._lower_bound


class MarkovianRegressor(object):
    """Class for all the regressors that are using within the Bayesian optimizer."""
    def __init__(self, Lipschitz_constant):
        """Lower and upper bounds are assumed to be 0.0 and 1.0"""
        # Store the parameters
        self._Lipschitz_constant = Lipschitz_constant

        # Set the value of the numerical accuracy
        # Used in the calculation of the kernel value close to the boundary
        # of the domain (0.0, 1.0)
        self._numerical_accuracy = 1e-3

        # Reset the (internally) stored samples
        self.reset_samples()

    def reset_samples(self) -> None:
        """Reset all samples and return the GP to initial state"""
        # shape = [n_samples, n_features]
        self._samples = np.array([])
        # shape = [n_samples, 1]
        self._values = np.array([])

    def add_sample(self, sample: float, sample_value: float) -> bool:
        """Add a sample to the internal buffer, returns True if successful"""
        # Make sure the sample and its value is at least a 2D array
        # in order to avoid errors when handling 1D arrays
        sample = np.atleast_2d(sample)
        sample_value = np.atleast_2d(sample_value)

        # Check if the sample is already observed
        if sample in self._samples:
            print(f'sample was already sampled --> {sample}')
            return False

        # Add the sample and sample value to the internal arrays
        try:
            self._samples = np.vstack((self._samples, sample))
            self._values = np.vstack((self._values, sample_value))
        except ValueError:
            self._samples = sample
            self._values = sample_value

        # Sort the observations such that x_1 < x_i < x_n
        sorting_ind = np.argsort(self._samples, axis=0)
        self._samples = np.take_along_axis(self._samples, sorting_ind, axis=0)
        self._values = np.take_along_axis(self._values, sorting_ind, axis=0)

        return True

    @property
    def number_of_samples(self) -> int:
        """Return the number of samples"""
        return self._samples.shape[0]

    def get_maximum_value(self) -> float:
        """Get the maximum value of the stored samples"""
        # Check if there are stored samples
        if self._values.size == 0:
            return 0.0
        else:
            # Get the maximum value
            return self._values.max()

    def predict(self, x: float) -> [Optional[float], Optional[float]]:
        """Return the mean and standard deviation for sample 'x'"""
        x = np.atleast_2d(x).T
        # Check if there are enough samples to predict
        if self.number_of_samples < 2:
            return [None, None]

        # Get the indices of the samples surrounding sample 'x'
        left_index = np.argmax(np.invert(self._samples <= x.T), axis=0) - 1
        right_index = left_index + 1

        # Adjust the indices for the bounds
        left_index[left_index == -1] = self.number_of_samples - 1
        right_index[right_index >= self.number_of_samples] = self.number_of_samples - 1

        # Get the surrounding samples
        sample_left = self._samples[left_index]
        sample_right = self._samples[right_index]

        # Get the values for the surrounding samples
        value_left = self._values[left_index]
        value_right = self._values[right_index]

        # Calculate the mean
        mu = (value_left * (sample_right - x) + value_right * (x - sample_left)) / (sample_right - sample_left)
        # Calculate the sigma
        sigma = np.sqrt(
            self._Lipschitz_constant**2 * ((sample_right - x) * (x - sample_left)) / (sample_right - sample_left)
        )
        sigma[np.isnan(sigma)] = 0.0  # correct for values where mean is zero

        return [mu, sigma]


class AcquisitionExpected(object):
    """Implementation of the expected improvement acquisition function"""
    def __init__(self, xi: float=0.0) -> None:
        # Initiate the parent class
        self._max_value = 0.0
        # Set the parameters
        self._xi = xi

    def compute(self, mu: float, sigma: float) -> float:
        """Return the expected improvement acquisition function values
        [brochu2010tutorial] page 13
        """
        with np.errstate(divide='ignore'):
            # correct for sigma values equal to zero
            boolean_selection = (sigma <= 0)
            sigma[boolean_selection] = 1e-9

            # Calculate the values
            Z = mu - (self._max_value + self._xi)
            Z_cdf = stats.norm.cdf(Z / sigma)
            Z_pdf = stats.norm.pdf(Z / sigma)
            acquisition_values = Z * Z_cdf + sigma * Z_pdf

            # correct for sigma values equal to zero
            acquisition_values[boolean_selection] = 0.0

        return acquisition_values

    def update_max_value(self, max_value: float) -> None:
        """Update the maximum found value of the samples"""
        if max_value is not None:
            self._max_value = max_value
