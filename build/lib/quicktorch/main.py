from .mlp import mlp

class quicktorch():
    def create_network(self, model_type, **kwargs):
        model_type = model_type.lower().strip()
        if model_type == 'fnn':
            return mlp(**kwargs)