from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax


@dataclass
class OptiParams:
    method: str = "LBFGS"
    max_iter: int = 100
    tol: float = 1e-6
    lr: float = 1e-3


def run_lbfgs(fun: Callable, x0: jax.Array, metaparams: OptiParams) -> tuple[jax.Array, jax.Array]:
    opt = optax.lbfgs()
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(grad, state, params, value=value, grad=grad, value_fn=fun)
        params = optax.apply_updates(params, updates)
        return (params, state)

    def cond(carry):
        params, state = carry
        grad = optax.tree.get(state, "grad")
        count = optax.tree.get(state, "count")
        err = optax.tree.norm(grad)
        jax.debug.print("LBFGS iter {}/{}: grad_norm={}", count, metaparams.max_iter, err)
        return (count == 0) | ((count < metaparams.max_iter) & (err >= metaparams.tol))

    init_state = opt.init(x0)
    final_params, final_state = jax.lax.while_loop(cond, step, (x0, init_state))
    return final_params, final_state


def run_adam(fun: Callable, x0: jax.Array, metaparams: OptiParams) -> tuple[jax.Array, jax.Array]:
    opt = optax.adam(metaparams.lr)
    opt_state = opt.init(x0)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(fun)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        grad_norm = jnp.linalg.norm(jax.flatten_util.ravel_pytree(grads)[0])
        return params, opt_state, loss, grad_norm

    params = x0
    for i in range(metaparams.max_iter):
        params, opt_state, loss, grad_norm = step(params, opt_state)
        jax.debug.print("Adam iter {}/{}: loss={:.4e}, grad_norm={:.4e}", i, metaparams.max_iter, loss, grad_norm)
        if grad_norm < metaparams.tol:
            break
    return params, opt_state


def run_adamw(fun: Callable, x0: jax.Array, metaparams) -> tuple[jax.Array, jax.Array]:
    """
    Runs AdamW optimization on the given function.

    Args:
        fun: Callable mapping params -> scalar loss.
        x0: Initial parameter array (can be PyTree).
        metaparams: Object with fields:
            - lr: learning rate
            - weight_decay: L2 regularization coefficient
            - max_iter: maximum iterations
            - tol: stopping tolerance on gradient norm

    Returns:
        (params, opt_state)

    """
    opt = optax.adamw(learning_rate=metaparams.lr, weight_decay=getattr(metaparams, "weight_decay", 0.0))
    opt_state = opt.init(x0)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(fun)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        grad_norm = jnp.linalg.norm(jax.flatten_util.ravel_pytree(grads)[0])
        return params, opt_state, loss, grad_norm

    params = x0
    for i in range(metaparams.max_iter):
        params, opt_state, loss, grad_norm = step(params, opt_state)
        jax.debug.print("AdamW iter {}/{}: loss={:.4e}, grad_norm={:.4e}", i, metaparams.max_iter, loss, grad_norm)
        if grad_norm < metaparams.tol:
            break

    return params, opt_state


def run_opt(
    fun: Callable,
    x0: jax.Array,
    metaparams: OptiParams | None = None,
):
    if metaparams is None:
        metaparams = OptiParams()
    if metaparams.method.lower() == "lbfgs":
        return run_lbfgs(fun, x0, metaparams)
    elif metaparams.method.lower() == "adam":
        return run_adam(fun, x0, metaparams)
    elif metaparams.method.lower() == "adamw":
        return run_adamw(fun, x0, metaparams)
    elif metaparams.method.lower() == "adam+lbfgs":
        final_params, final_state = run_adam(fun, x0, metaparams)
        return run_lbfgs(fun, final_params, metaparams)
    else:
        raise ValueError(f"Unknown optimization method: {metaparams.method}")
