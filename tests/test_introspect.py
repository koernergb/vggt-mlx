from tools.introspect import _constructor_default


class ExampleArchitecture:
    def __init__(self, init_values=0.01):
        # Simulate a trained value that must not be reported as initialization.
        self.gamma = -2.7e-5


def test_constructor_default_is_not_confused_with_trained_parameter():
    architecture = ExampleArchitecture()
    assert _constructor_default(architecture, "init_values") == 0.01
