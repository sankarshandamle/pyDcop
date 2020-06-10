"""

Tests for loading DCOP yaml files.

These tests check for correct parsing of the DCOP definition files.

"""
import unittest
import pathlib
from pydcop.dcop.yamldcop import load_dcop_from_file, load_dcop
from pydcop.dcop.objects import (
    ContinuousDomain,
    Domain,
)
from pydcop.dcop.relations import (
    NAryFunctionRelation,
)
from pydcop.utils.expressionfunction import ExpressionFunction

dcop_test_str = """
    name: 'dcop test'
    description: 'Testing of DCOP yaml parsing'
    objective: min

    domains: 
        dint:
            values  : [0, 1, 2, 3, 4]
            type   : non_semantic
        dstr:
            values  : ['A', 'B', 'C', 'D', 'E']
        dcont:
            values  : [0 .. 1]
            type   : non_semantic
            initial_value: 3
        dbool:
            values  : [true, false]

    variables:
        var1:
            domain: dint
            initial_value: 0
            yourkey: yourvalue
            foo: bar
        var2:
            domain: dstr
            initial_value: 'A'
        var3:
            domain: dint
            initial_value: 0
            cost_function: var3 * 0.5
        var4:
            domain: dcont
            initial_value: 0
            cost_function: var4 * 0.6
    
    external_variables:
        ext_var1:
            domain: dbool
            initial_value: False

    constraints:
        c1_intention:
            type: intention
            function: var3 - var1
        c2_intention_source: 
            type: intention
            source: "branin_constraint.py"
            function: source.branin(var1, var2)
        c3_intention_property: 
            type: intention
            source: "bird_constraint.py"
            function: source.bird(var1, var2)
            properties:
                Lipschitz_constant: 1326.5

    agents:
        a1:
            capacity: 100
        a2:
            capacity: 100
        a3:
            capacity: 100
        a4:
            capacity: 100
"""

class TestDomains(unittest.TestCase):
    def load_dcop(self):
        # Get the path to the tests/instances directory
        current_dir = pathlib.Path(__file__).parent.absolute()
        main_dir = current_dir.parent / 'instances'

        # Load the DCOP
        dcop = load_dcop(
            dcop_test_str, 
            main_dir=main_dir,
        )

        return dcop

    def test_classes(self):
        dcop = self.load_dcop()
        # Check the classes
        self.assertIsInstance(dcop.domains['dint'], Domain)
        self.assertIsInstance(dcop.domains['dstr'], Domain)
        self.assertIsInstance(dcop.domains['dcont'], ContinuousDomain)
        self.assertIsInstance(dcop.domains['dbool'], Domain)

    def test_values(self):
        dcop = self.load_dcop()
        # Check the values
        self.assertEqual(dcop.domains['dint'].values, (0, 1, 2, 3, 4))
        self.assertEqual(dcop.domains['dstr'].values, ('A', 'B', 'C', 'D', 'E'))
        self.assertEqual(dcop.domains['dcont'].values, (0.0, 1.0))
        self.assertEqual(dcop.domains['dbool'].values, (True, False))

    def test_bounds(self):
        dcop = self.load_dcop()
        # Check the bounds
        self.assertEqual(dcop.domains['dcont'].lower_bound, 0.0)
        self.assertEqual(dcop.domains['dcont'].upper_bound, 1.0)

class TestConstraints(unittest.TestCase):
    def load_dcop(self):
        # Get the path to the tests/instances directory
        current_dir = pathlib.Path(__file__).parent.absolute()
        main_dir = current_dir.parent / 'instances'

        # Load the DCOP
        dcop = load_dcop(
            dcop_test_str, 
            main_dir=main_dir,
        )

        return dcop

    def test_intentional(self):
        dcop = self.load_dcop()

        # Check the constraints
        c1_intention = dcop.constraints['c1_intention']
        self.assertIsInstance(c1_intention, NAryFunctionRelation)
        self.assertIsInstance(c1_intention.function, ExpressionFunction)
        self.assertIn('var1', c1_intention.scope_names)
        self.assertIn('var3', c1_intention.scope_names)
        self.assertEqual(c1_intention.expression, 'var3 - var1')
        self.assertIsNone(c1_intention.function._source_file)

        c2_intention_source = dcop.constraints['c2_intention_source']
        self.assertIsInstance(c2_intention_source, NAryFunctionRelation)
        self.assertIsInstance(c2_intention_source.function, ExpressionFunction)
        self.assertIn('var1', c2_intention_source.scope_names)
        self.assertIn('var2', c2_intention_source.scope_names)
        self.assertEqual(c2_intention_source.expression, 'source.branin(var1, var2)')

        current_dir = pathlib.Path(__file__).parent.absolute()
        main_dir = current_dir.parent / 'instances'
        self.assertEqual(c2_intention_source.function._source_file, main_dir / 'branin_constraint.py')

        c3_intention_property = dcop.constraints['c3_intention_property']
        self.assertIsInstance(c3_intention_property, NAryFunctionRelation)
        self.assertIsInstance(c3_intention_property.function, ExpressionFunction)
        self.assertIn('var1', c3_intention_property.scope_names)
        self.assertIn('var2', c3_intention_property.scope_names)
        self.assertEqual(c3_intention_property.expression, 'source.bird(var1, var2)')
        self.assertEqual(c3_intention_property._properties, {'Lipschitz_constant': 1326.5})

TestConstraints().test_intentional()