import jax
import jax.random as jrandom
def JaxRNG():
    def __init__(self, seed):
        self.key = jrandom.PRNGKey(seed)
    def get_key(self):
        self.key, tmp = jrandom.split(self.key, 2)
        return tmp 
    

@jax.jit
def soft_update_params(current_state, target_state, tau):
    target_state.params = jax.tree_map(lambda x, y: tau*y + x*(1-tau), current_state.params, target_state.params)
    return target_state

