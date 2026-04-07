(import jax
        jax [lax]
        hy.pyops [reduce +]
        hy.models [List :as HyList
                   Expression
                   Symbol])



;;; Helper functions & macros

(defn find-symbols [expr]
  (match expr
    [] []
    (Expression [head #* tail]) (find-symbols tail) ; first element in Expression is special
    [head] (find-symbols head)
    [head #* tail] (+ (find-symbols head) (find-symbols tail))
    (Symbol) [expr]
    _ []))                       

(defn unique [lst] (list (set lst)))

(defn fn-with-bindings [bindings #* body]
  (let [binding-list (hy.gensym "binding-list")]
    `(fn [~binding-list]
       (let [~bindings ~binding-list]
         ~@body))))

(defn _lcond-form [bindings pred true-case false-case]
  "Build a (lax.cond ...) expression with an if-like surface syntax."
  `(hy.I.jax.lax.cond ~pred
                      ~(fn-with-bindings bindings true-case)
                      ~(fn-with-bindings bindings false-case)
                      ~bindings))

(defmacro if/lp [bindings pred true-case false-case]
  "Explicit-binding lax.cond wrapper. Lower-level form of if/l.

  The bindings list is the explicit closure passed through lax.cond;
  use this when you don't want auto-detection of free symbols.

      (if/lp [a b] (= a b) (+ a 1) (- b 2))
  "
  (_lcond-form bindings pred true-case false-case))

(defn _cond/l [args]
  (if (> (len args) 1)
    `(if/l ~(get args 0)
      ~(get args 1)
      ~(_cond/l (cut args 2 None)))
    (get args 0)))

;;; API Macros

(defmacro if/l [pred true-case false-case]
  "if-like wrapper around lax.cond with auto binding detection"
  (let [bindings (unique (reduce + (map find-symbols [pred true-case false-case])))]
    (_lcond-form bindings pred true-case false-case)))

(defmacro cond/l [#* args]
  "cond-like chain of lax.cond branches; takes an odd number of forms
  (pred1 result1 pred2 result2 ... default)"
  (if (not (% (len args) 2))
    (raise (ValueError "cond/l requires an odd number of arguments"))
    (_cond/l args)))


(defmacro defn/j [#* args]
  "define jit-compiled function; replacing control flow with lax constructs"
  (match args
    ; plain defn
    [(| (Symbol name) (Expression name)) (HyList args) #* body]
    `(defn [hy.I.jax.jit] ~name ~args ~@body)

    ; with decorators
    [(HyList decorators) (| (Symbol name) (Expression name)) (HyList args) #* body]
    `(defn [hy.I.jax.jit ~@decorators] ~name ~args ~@body)))

(defmacro mapv [f v]
  `((hy.I.jax.vmap ~f) ~v))

(defmacro while/lp [binding cond-expr body-expr]
  "Explicit-binding lax.while_loop wrapper. Lower-level form of while/l.

  The state binding is [pattern init]; the body is a single expression
  that returns the new state. The pattern may destructure a tuple.

      (while/lp [v 0]
                (< v 10)
                (+ v 1))

      (while/lp [[a b] #(0 1)]
                (< a 10)
                #((+ a 1) (* b 2)))

  Use this when you want full control over what's in the carry, e.g.
  to keep loop-local temporaries out of it, or when the body is more
  naturally an expression than a sequence of (setv ...) statements.
  "
  (when (!= (len binding) 2)
    (raise (ValueError "while/lp binding must be [pattern init]")))
  (let [pattern (get binding 0)
        init    (get binding 1)
        carry   (hy.gensym "carry")]
    `(hy.I.jax.lax.while_loop
       (fn [~carry] (setv ~pattern ~carry) ~cond-expr)
       (fn [~carry] (setv ~pattern ~carry) ~body-expr)
       ~init)))

(defmacro fori/lp [state-binding index-binding body-expr]
  "Explicit-binding lax.fori_loop wrapper. Lower-level form of fori/l.

  Bindings-first ordering: carry state leads, iteration descriptor
  follows.

      (fori/lp [acc 0]    ; carried state: pattern + init
               [i 0 n]    ; index name, lower, upper
               (+ acc i))

  Like while/lp, the state pattern may destructure a tuple.
  "
  (when (!= (len state-binding) 2)
    (raise (ValueError "fori/lp state binding must be [pattern init]")))
  (when (!= (len index-binding) 3)
    (raise (ValueError "fori/lp index binding must be [i lower upper]")))
  (let [pattern (get state-binding 0)
        init    (get state-binding 1)
        i       (get index-binding 0)
        lower   (get index-binding 1)
        upper   (get index-binding 2)
        carry   (hy.gensym "carry")]
    `(hy.I.jax.lax.fori_loop
       ~lower ~upper
       (fn [~i ~carry] (setv ~pattern ~carry) ~body-expr)
       ~init)))

(defn _check-carry-names [carry ctx]
  (when (or (not (isinstance carry HyList)) (= (len carry) 0))
    (raise (ValueError (+ ctx " carry must be a non-empty [name ...] list"))))
  (for [n carry]
    (when (not (isinstance n Symbol))
      (raise (ValueError (+ ctx " carry names must be plain symbols"))))))

(defmacro while/l [carry cond-expr body-expr]
  "lax.while_loop wrapper with auto-init from the enclosing scope.

  `carry` is an explicit list of state variable *names*; each name
  must already be bound in the surrounding scope, and those values
  become the loop's initial state. `cond-expr` and `body-expr` are
  evaluated with those names destructured from the carry; `body-expr`
  is a single expression that evaluates to the new carry.

      (setv v 0)
      (setv v (while/l [v] (< v 10) (+ v 1)))   ; single-state

      (setv a 0  b 1)
      (setv [a b] (while/l [a b]                 ; multi-state
                           (< a 100)
                           #(b (+ a b))))

  With one carry name, the init and body are scalars; with N > 1,
  they are N-tuples in the declared order.
  "
  (_check-carry-names carry "while/l")
  (let [names (list carry)
        param (hy.gensym "carry")]
    (if (= (len names) 1)
      (let [n (get names 0)]
        `(hy.I.jax.lax.while_loop
           (fn [~param] (setv ~n ~param) ~cond-expr)
           (fn [~param] (setv ~n ~param) ~body-expr)
           ~n))
      `(hy.I.jax.lax.while_loop
         (fn [~param] (setv [~@names] ~param) ~cond-expr)
         (fn [~param] (setv [~@names] ~param) ~body-expr)
         #(~@names)))))

(defmacro fori/l [carry index-binding body-expr]
  "lax.fori_loop wrapper with auto-init from the enclosing scope.

  `carry` is an explicit list of state variable *names* (same as
  `while/l`); their initial values come from the enclosing scope.
  `index-binding` is `[i (range ...)]` where the iterable is a
  literal (range n) or (range lo hi).

  Bindings-first ordering matches Clojure's `loop` and CL `do`: the
  state carry is what the body produces a new version of, and the
  iteration descriptor is secondary.

      (setv acc 0)
      (setv acc (fori/l [acc] [i (range n)] (+ acc i)))

      (setv a 0  b 1)
      (setv [a b] (fori/l [a b] [i (range n)] #(b (+ a b))))
  "
  (_check-carry-names carry "fori/l")
  (when (!= (len index-binding) 2)
    (raise (ValueError "fori/l index binding must be [i (range ...)]")))
  (let [i-sym      (get index-binding 0)
        range-expr (get index-binding 1)]
    (when (not (and (isinstance range-expr Expression)
                    (> (len range-expr) 0)
                    (isinstance (get range-expr 0) Symbol)
                    (= (str (get range-expr 0)) "range")))
      (raise (ValueError "fori/l iterable must be a (range ...) form")))
    (let [range-args (cut range-expr 1 None)
          names (list carry)
          param (hy.gensym "carry")
          lower None
          upper None]
      (cond
        (= (len range-args) 1)
        (do (setv lower 0) (setv upper (get range-args 0)))
        (= (len range-args) 2)
        (do (setv lower (get range-args 0))
            (setv upper (get range-args 1)))
        True
        (raise (ValueError
                 "fori/l only supports (range upper) or (range lower upper)")))
      (if (= (len names) 1)
        (let [n (get names 0)]
          `(hy.I.jax.lax.fori_loop
             ~lower ~upper
             (fn [~i-sym ~param] (setv ~n ~param) ~body-expr)
             ~n))
        `(hy.I.jax.lax.fori_loop
           ~lower ~upper
           (fn [~i-sym ~param] (setv [~@names] ~param) ~body-expr)
           #(~@names))))))

;;; lax.scan bindings

(defmacro scan/lp [state-binding iter-binding body-expr]
  "Explicit-binding lax.scan wrapper. Lower-level form of scan/l.

  `state-binding` is [pattern init] for the carry; `iter-binding` is
  [x xs] where xs is the stacked input. The body evaluates to a
  #(new-carry y) pair, and scan/lp returns #(final-carry stacked-ys).

      (scan/lp [acc (jnp.int32 0)]
               [x xs]
               #((+ acc x) (+ acc x)))
  "
  (when (!= (len state-binding) 2)
    (raise (ValueError "scan/lp state binding must be [pattern init]")))
  (when (!= (len iter-binding) 2)
    (raise (ValueError "scan/lp iter binding must be [x xs]")))
  (let [pattern (get state-binding 0)
        init    (get state-binding 1)
        x-sym   (get iter-binding 0)
        xs-expr (get iter-binding 1)
        carry   (hy.gensym "carry")]
    `(hy.I.jax.lax.scan
       (fn [~carry ~x-sym]
         (setv ~pattern ~carry)
         ~body-expr)
       ~init
       ~xs-expr)))

(defmacro scan/l [carry iter-binding body-expr]
  "lax.scan wrapper with auto-init from the enclosing scope.

  `carry` lists the state variable *names*; their initial values come
  from the enclosing scope. `iter-binding` is [x xs] where xs is the
  stacked input. The body evaluates to a #(new-carry y) pair.
  scan/l returns #(final-carry stacked-ys).

      ; cumulative sum
      (setv acc (jnp.int32 0))
      (setv [final sums]
            (scan/l [acc] [x xs] #((+ acc x) (+ acc x))))

      ; Fibonacci sequence
      (setv a (jnp.int32 0)  b (jnp.int32 1))
      (setv [_ seq]
            (scan/l [a b] [x (jnp.arange n)] #(#(b (+ a b)) a)))
  "
  (_check-carry-names carry "scan/l")
  (when (!= (len iter-binding) 2)
    (raise (ValueError "scan/l iter binding must be [x xs]")))
  (let [names (list carry)
        x-sym   (get iter-binding 0)
        xs-expr (get iter-binding 1)
        param   (hy.gensym "carry")]
    (if (= (len names) 1)
      (let [n (get names 0)]
        `(hy.I.jax.lax.scan
           (fn [~param ~x-sym] (setv ~n ~param) ~body-expr)
           ~n
           ~xs-expr))
      `(hy.I.jax.lax.scan
         (fn [~param ~x-sym] (setv [~@names] ~param) ~body-expr)
         #(~@names)
         ~xs-expr))))

;;; lax.switch binding

(defmacro case/l [idx #* branches]
  "lax.switch wrapper with auto-detected captured bindings.

  Like cond/l but branches are selected by an integer index rather
  than predicates. Free symbols in idx and the branch expressions are
  auto-detected and closed over.

      (case/l idx
              (* x 2)      ; branch 0
              (+ x 1)      ; branch 1
              (- x 1))     ; branch 2
  "
  (when (= (len branches) 0)
    (raise (ValueError "case/l requires at least one branch")))
  (let [bindings (unique (reduce + (map find-symbols (+ [idx] (list branches)))))
        branch-fns (lfor b branches (fn-with-bindings bindings b))]
    `(hy.I.jax.lax.switch ~idx [~@branch-fns] ~bindings)))

;;; lax.associative_scan and lax.map

(defmacro ascan/l [f xs #* rest]
  "lax.associative_scan wrapper. `f` must be a binary associative fn.

      (ascan/l + xs)                  ; prefix sums
      (ascan/l + xs :reverse True)    ; extra kwargs are spliced through
  "
  `(hy.I.jax.lax.associative_scan ~f ~xs ~@rest))

(defmacro mapl [f xs]
  "lax.map wrapper: sequential map (vs mapv which is parallel vmap).

      (mapl f xs)
  "
  `(hy.I.jax.lax.map ~f ~xs))

;;; jax.tree.map binding

(defmacro mapt [f #* trees]
  "jax.tree.map wrapper: map `f` over the leaves of one or more
  pytrees (corresponding leaves when multiple trees are given).

      (mapt (fn [p g] (- p (* lr g))) params grads)
  "
  `(hy.I.jax.tree.map ~f ~@trees))

;;; jax.jvp and jax.vjp bindings

(defmacro jvp/l [primals tangents f]
  "jax.jvp wrapper.

  `primals` and `tangents` are bracketed lists of expressions that
  become the positional primal and tangent tuples passed to jax.jvp.
  `f` is a function of the same arity. Returns #(primal-out tangent-out).

      (setv [y dy] (jvp/l [x] [1.0] (fn [a] (* a a))))
  "
  (when (not (isinstance primals HyList))
    (raise (ValueError "jvp/l primals must be a [expr ...] list")))
  (when (not (isinstance tangents HyList))
    (raise (ValueError "jvp/l tangents must be a [expr ...] list")))
  (when (!= (len primals) (len tangents))
    (raise (ValueError "jvp/l primals and tangents must have the same length")))
  `(hy.I.jax.jvp ~f #(~@primals) #(~@tangents)))

(defmacro vjp/l [primals f]
  "jax.vjp wrapper.

  `primals` is a bracketed list of expressions passed as the
  positional primal arguments to `f`. Returns #(output vjp-fn); call
  vjp-fn on a cotangent to get a tuple of cotangents w.r.t. primals.

      (setv [y vjp-fn] (vjp/l [x] (fn [a] (* a a))))
      (setv [dx] (vjp-fn 1.0))
  "
  (when (not (isinstance primals HyList))
    (raise (ValueError "vjp/l primals must be a [expr ...] list")))
  `(hy.I.jax.vjp ~f ~@primals))

;;; PRNG key splitting sugar

(defmacro with-keys [key subs #* body]
  "Split `key` into fresh sub-keys and rebind.

  Produces a setv that consumes `key`, rebinds it to the first slot
  of the split, and binds each name in `subs` to the subsequent
  slots. The rebinding escapes into the enclosing scope (matching
  how Python JAX code threads keys) so subsequent code sees the
  advanced key.

      (setv key (jax.random.PRNGKey 0))
      (with-keys key [k1 k2 k3]
        (setv x (jax.random.normal k1 #(4)))
        (setv y (jax.random.normal k2 #(4)))
        (setv z (jax.random.normal k3 #(4))))
  "
  (when (not (isinstance subs HyList))
    (raise (ValueError "with-keys sub-key list must be a [name ...] list")))
  (for [n subs]
    (when (not (isinstance n Symbol))
      (raise (ValueError "with-keys sub-key names must be plain symbols"))))
  (let [n (+ 1 (len subs))]
    `(do
       (setv [~key ~@subs] (hy.I.jax.random.split ~key ~n))
       ~@body)))
