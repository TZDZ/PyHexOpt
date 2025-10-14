from collections.abc import Callable

import jax
import optax


def run_opt(
    fun: Callable,
    x0: jax.Array,
    method: str = "LBFGS",
    max_iter: int = 50,
    tol: float = 1e-6,
):
    if method == "LBFGS":
        opt = optax.lbfgs()
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(grad, state, params, value=value, grad=grad, value_fn=fun)
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = optax.tree.get(state, "count")
        grad = optax.tree.get(state, "grad")
        err = optax.tree.norm(grad)
        jax.debug.print("Iteration {}/{}: err={}", iter_num, max_iter, err)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (x0, opt.init(x0))
    final_params, final_state = jax.lax.while_loop(continuing_criterion, step, init_carry)
    return final_params, final_state
