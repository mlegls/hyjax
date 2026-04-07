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

(defmacro lcond [bindings pred true-case false-case]
  "lax cond, with if-like syntax"
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

(defmacro while/l [binding cond-expr body-expr]
  "lax.while_loop wrapper with a let-style state binding.

  (while/l [v 0]
           (< v 10)
           (+ v 1))

  The binding's left-hand side may be any pattern that Hy's (setv ...)
  accepts, so multi-state loops can destructure a tuple:

  (while/l [[a b] #(0 1)]
           (< a 10)
           #((+ a 1) (* b 2)))
  "
  (when (!= (len binding) 2)
    (raise (ValueError "while/l binding must be [pattern init]")))
  (let [pattern (get binding 0)
        init    (get binding 1)
        carry   (hy.gensym "carry")]
    `(hy.I.jax.lax.while_loop
       (fn [~carry] (setv ~pattern ~carry) ~cond-expr)
       (fn [~carry] (setv ~pattern ~carry) ~body-expr)
       ~init)))

(defmacro fori/l [index-binding state-binding body-expr]
  "lax.fori_loop wrapper.

  (fori/l [i 0 n]    ; index name, lower, upper
          [acc 0]    ; carried state
          (+ acc i))

  Like while/l, the state pattern may destructure a tuple.
  "
  (when (!= (len index-binding) 3)
    (raise (ValueError "fori/l index binding must be [i lower upper]")))
  (when (!= (len state-binding) 2)
    (raise (ValueError "fori/l state binding must be [pattern init]")))
  (let [i       (get index-binding 0)
        lower   (get index-binding 1)
        upper   (get index-binding 2)
        pattern (get state-binding 0)
        init    (get state-binding 1)
        carry   (hy.gensym "carry")]
    `(hy.I.jax.lax.fori_loop
       ~lower ~upper
       (fn [~i ~carry] (setv ~pattern ~carry) ~body-expr)
       ~init)))
