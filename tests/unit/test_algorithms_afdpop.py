"""

Tests for the AF-DPOP algorithm and its custom classes.

"""
import unittest
from pydcop.dcop.relations import (
    NAryFunctionRelation,
    NAryMatrixRelation,
)
from pydcop.dcop.objects import (
    ContinuousDomain,
    Domain,
    Variable,
)
from pydcop.algorithms.afdpop import (
    InterpolatedMatrixRelation
)


class TestRelations(unittest.TestCase):

    def test_matrix_to_functional_relation(self):
        # Create the variables
        x1 = Variable('x1', [5, 6, 7])
        x2 = Variable('x2', [1, 2])

        # Create a matrix relation
        relation_values = [[2, 16],
                           [4, 32],
                           [8, 64]]
        matrix_relation = NAryMatrixRelation(
            variables=[x1, x2], 
            matrix=relation_values, 
            name='matrix relation',
        )

        # Convert into a matrix relation
        functional_relation = InterpolatedMatrixRelation(matrix_relation)
        
        # Check the exact values
        self.assertEqual(functional_relation.get_value_for_assignment({'x1': 5, 'x2': 1}), 2.0)
        self.assertEqual(functional_relation.get_value_for_assignment({'x1': 6, 'x2': 1}), 4.0)
        self.assertEqual(functional_relation.get_value_for_assignment({'x1': 7, 'x2': 1}), 8.0)

        self.assertEqual(functional_relation.get_value_for_assignment({'x1': 5, 'x2': 2}), 16.0)
        self.assertEqual(functional_relation.get_value_for_assignment({'x1': 6, 'x2': 2}), 32.0)
        self.assertEqual(functional_relation.get_value_for_assignment({'x1': 7, 'x2': 2}), 64.0)

        # Check selected interpolated values
        self.assertEqual(functional_relation.get_value_for_assignment({'x1': 5.5, 'x2': 1}), 3.0)
        self.assertEqual(functional_relation.get_value_for_assignment({'x1': 6.5, 'x2': 1}), 6.0)

        self.assertEqual(functional_relation.get_value_for_assignment({'x1': 5.5, 'x2': 2}), 24.0)
        self.assertEqual(functional_relation.get_value_for_assignment({'x1': 6.5, 'x2': 2}), 48.0)

        self.assertEqual(functional_relation.get_value_for_assignment({'x1': 5, 'x2': 1.5}), 9.0)
        self.assertEqual(functional_relation.get_value_for_assignment({'x1': 6, 'x2': 1.5}), 18.0)
        self.assertEqual(functional_relation.get_value_for_assignment({'x1': 7, 'x2': 1.5}), 36.0)