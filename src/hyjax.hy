(import ;jax 
        jax [numpy :as jnp
             lax
             grad jit vmap
             random]
        hy [macroexpand]
        hy.pyops [reduce +]
        hy.models [List :as HyList 
                   Expression
                   Symbol])

(require hyrule [with-gensyms smacrolet])

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
  (with-gensyms [binding-list]
    `(fn [binding-list]
       (let [~bindings binding-list]
         ~@body))))

(defmacro lcond [bindings pred true-case false-case]
  "lax cond, with if-like syntax"
  `(lax.cond ~pred
             ~(fn-with-bindings bindings true-case)
             ~(fn-with-bindings bindings false-case)
             ~bindings))

(defn _cond/j [args]
  (if (> (len args) 1)
    `(if/j ~(get args 0)
      ~(get args 1)
      ~(_cond/j (cut args 2 None)))
    (get args 0)))

;;; API Macros

(defmacro if/j [pred true-case false-case]
  "lcond with auto binding detection"
  (let [bindings (unique (reduce + (map find-symbols [pred true-case false-case])))]
    `(lcond ~bindings ~pred ~true-case ~false-case)))

(defmacro cond/j [#* args]
  "cond macro with lax cond"
  (if (not (% (len args) 2))
    (raise (ValueError "cond/j requires an odd number of arguments"))
    (_cond/j args)))


(defmacro defn/j [#* args]
  "define jit-compiled function; replacing control flow with lax constructs"
  (match args
    ; plain defn 
    [(| (Symbol name) (Expression name)) (HyList args) #* body] 
    `(defn [jit] ~name ~args ~@body)

    ; with decorators
    [(HyList decorators) (| (Symbol name) (Expression name)) (HyList args) #* body] 
    `(defn [jit ~@decorators] ~name ~args ~@body)))

(defmacro mapv [f v]
  `((vmap ~f) ~v))
