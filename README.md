# Hyjax

```bash
uv add hyjax         # or: pip install hyjax
```

Dev setup:

```bash
git clone https://github.com/mlegls/hyjax
cd hyjax
uv sync
uv run pytest                       # run the test suite
uv run hy examples/experiments.hy   # run the example script
```

Hyjax is [Hy](https://github.com/hylang/hy) bindings for [JAX](https://github.com/google/jax). 


> When walking about the countryside of Italy, the people will not hesitate to tell you that JAX has [“una anima di pura programmazione funzionale”](https://www.sscardapane.it/iaml-backup/jax-intro/).
> –– <cite>[🔪 JAX - The Sharp Bits 🔪](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html?highlight=pure%20functional)</cite>

> “As one would expect from its goals, artificial intelligence research generates many significant programming problems. In other programming cultures this spate of problems spawns new languages. ... We toast the Lisp programmer who pens his thoughts within nests of parentheses.”
> -- <cite>Alan J. Perlis</cite>

> You now know enough to be dangerous with Hy. You may now smile villainously and sneak off to your Hydeaway to do unspeakable things.
> –– <cite>[Hy Tutorial](https://docs.hylang.org/en/stable/tutorial.html?highlight=hydeaway#next-steps)</cite>

The goals are
1. the name pun was funny
2. let fully JIT-compiled JAX code be written in idiomatic Lisp

Some examples loosely inspired by the [JAX Quickstart](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) tutorials are in `examples/experiments.hy`. The [Hy](https://docs.hylang.org/en/stable/index.html) and [JAX](https://jax.readthedocs.io/en/latest/) docs might also be helpful.

## Features
- [x] `(defn/j f [args] body)` as `@jit def f(args): ...`
  - `/j` for **j**it; identical syntax to Hy's [`defn`](https://docs.hylang.org/en/stable/api.html#defn); supports other decorators, annotations, variadic & keyword args
- [x] `(mapv f vec)` as `vmap(f)(vec)`
- [x] `(if/l pred then else)` as `lax.cond`
  - `/l` for **l**ax; if-like binary form
- [x] `(cond/l pred1 result1 pred2 result2 ... default)` as a chain of `lax.cond`
  - multi-branch form, mirroring Lisp `cond`
- [x] `(if/lp [bindings] pred then else)` — explicit-binding ("lax pure") form of `if/l`; you list the closed-over symbols yourself instead of relying on auto-detection
- [x] `(while/l [carry...] cond body-expr)` as `lax.while_loop`, in
  pure-expression style. Carry *names* are listed explicitly and
  their *initial values* are taken from the enclosing scope. The
  body is a single expression that evaluates to the new carry (a
  scalar for one name, an N-tuple for N > 1). The loop form itself
  returns the final carry.
  ```hy
  ; single-state
  (setv v 0)
  (setv v (while/l [v] (< v 10) (+ v 1)))

  ; multi-state
  (setv a 0  b 1)
  (setv [a b] (while/l [a b] (< a 100) #(b (+ a b))))
  ```
- [x] `(fori/l [carry...] [i (range ...)] body-expr)` as
  `lax.fori_loop`, same style. Bindings-first ordering matches
  Clojure's `loop` and CL `do` — the state carry leads, and the
  iteration descriptor follows.
  ```hy
  (setv acc 0)
  (setv acc (fori/l [acc] [i (range n)] (+ acc i)))

  (setv a 0  b 1)
  (setv [a b] (fori/l [a b] [i (range n)] #(b (+ a b))))
  ```
- [x] `(while/lp [pattern init] cond body-expr)` and
  `(fori/lp [pattern init] [i lo hi] body-expr)` — explicit-binding
  ("lax pure") forms of the loops. Use these when you want full
  control over what's in the carry, e.g. to keep loop-local
  temporaries out of it, or when you want to initialize with a
  different expression than a same-named outer binding.
- [x] `(scan/l [carry...] [x xs] body-expr)` as `lax.scan`. Body
  evaluates to a `#(new-carry y)` pair; the form returns
  `#(final-carry stacked-ys)`. Same carry-from-scope convention as
  `while/l` / `fori/l`. `scan/lp` is the explicit `[pattern init]`
  variant.
  ```hy
  ; cumulative sum
  (setv acc 0)
  (setv [final sums]
        (scan/l [acc] [x xs] #((+ acc x) (+ acc x))))
  ```
- [x] `(case/l idx expr0 expr1 ...)` as `lax.switch`. Like `cond/l`
  but branches are selected by integer index; free symbols are
  auto-detected and closed over.
- [x] `(ascan/l f xs)` as `lax.associative_scan`. Extra kwargs are
  spliced through: `(ascan/l + xs :reverse True)`.
- [x] `(mapl f xs)` as `lax.map` (sequential), complementing `mapv`
  which is `jax.vmap`.
- [x] `(mapt f tree ...)` as `jax.tree.map`. Accepts multiple
  pytrees for co-map.
  ```hy
  (mapt (fn [p g] (- p (* lr g))) params grads)
  ```
- [x] `(jvp/l [primals] [tangents] f)` as `jax.jvp`. Returns
  `#(primal-out tangent-out)`.
  ```hy
  (setv [y dy] (jvp/l [(jnp.float32 3.0)] [(jnp.float32 1.0)]
                      (fn [a] (* a a))))
  ```
- [x] `(vjp/l [primals] f)` as `jax.vjp`. Returns `#(output vjp-fn)`.
  ```hy
  (setv [y vjp-fn] (vjp/l [x] (fn [a] (* a a a))))
  (setv [dx]       (vjp-fn 1.0))
  ```
- [x] `(with-keys key [k1 k2 k3] body)` — PRNG key-splitting sugar.
  Consumes `key`, rebinds it to `split(key, 4)[0]`, and binds `k1`,
  `k2`, `k3` to the remaining slots. The rebinding escapes into the
  enclosing scope so subsequent code sees the advanced key (matches
  how Python JAX code threads keys).
  ```hy
  (setv key (jax.random.PRNGKey 0))
  (with-keys key [k1 k2]
    (setv x (jax.random.normal k1 #(4)))
    (setv y (jax.random.normal k2 #(4))))
  ```

### Naming convention

- `/j` — **j**it. Currently just `defn/j`.
- `/l` — **l**ax, in its idiomatic form. Always an expression.
  For `if/l` / `cond/l`, free symbols in the branches are
  auto-detected and closed over. For `while/l` / `fori/l`, you list
  the carry names explicitly and their initial values are pulled
  from the enclosing scope.
- `/lp` — **l**ax **p**ure. Lower-level explicit forms: for
  `if/lp`, you list the closed-over bindings yourself; for
  `while/lp` / `fori/lp`, you list the carry names *and* their
  initial values (`[name init]`). Use these when the `/l` sugar is
  inconvenient — e.g. initializing with a different expression than
  a same-named outer binding.

