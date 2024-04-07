import umbridge

class Gaussian1D(umbridge.Model):

    def __init__(self):
        super().__init__("simulation_model")
        self._mean = 0
        self._std = 1

    def get_input_sizes(self, config):
        return [1]

    def get_output_sizes(self, config):
        return [1]

    def __call__(self, parameters, config):
        diff = parameters[0][0] - self._mean
        logp = - 0.5 / self._std**2 * diff**2
        return [[logp]]

    def supports_evaluate(self):
        return True


model = Gaussian1D()

umbridge.serve_models([model], 4244)