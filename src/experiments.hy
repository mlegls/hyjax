(import ;jax 
        jax [numpy :as jnp
             lax
             grad jit vmap
             random]
        hy [macroexpand]
        hyrule [macroexpand-all]
        hy.pyops [reduce +]
        hy.models [List :as HyList 
                   Expression
                   Symbol])

(require hyrule [with-gensyms smacrolet]
         hyjax [defn/j mapv lcond if/j cond/j])

;;; Experiments

(setv key (random.PRNGKey 0))

(print (random.normal key #(10)))

(let [size 3000
      x (random.normal key #(size size) :dtype jnp.float32)]
  (print (. jnp (dot x x.T) (block_until_ready))))

(defn/j selu [x [alpha 1.67] [lmbda 1.05]]
        (* lmbda (jnp.where (> x 0) x (- (* alpha (jnp.exp x) alpha)))))

(let [x (random.normal key #(1000000))]
  (print (. (selu x) (block_until_ready))))

(let [mat (random.normal key #(150 100))
      batched_x (random.normal key #(10 100))]
  (defn/j apply_matrix [v]
          (jnp.dot mat v))
  (defn/j mapv-apply-matrix [v_batched]
          (mapv apply_matrix v_batched))
  (print (. (mapv-apply-matrix batched_x) (block_until_ready))))

(defn/j sum_logistic [x]
        (jnp.sum (/ 1.0 (+ 1.0 (jnp.exp (- x))))))

(let [x_small (jnp.arange 3.0)
      derivative_fn (grad sum_logistic)]
  (print (derivative_fn x_small)))

(defn/j test-if [x]
        (if/j (< x 3)
              (* 3 (** x 2))
              (* -4 x)))   

(print (test-if 2))

(defn/j test-nested-if [x]
        (if/j (< x 3) (* 3 (** x 2))
              (if/j (< x 5) (* -4 x)
                    (* 5 x))))

(print (test-nested-if 2))

(defn/j test-cond [x]
        (cond/j
          (< x 3) (* 3 (** x 2))
          (< x 5) (* -4 x)
          (* 5 x)))

(print (test-cond 2))


; TODO fix mismatched branch structure
(defn/j test-cond [x]
        (let [operand (jnp.array [0])]
          (cond/j False (+ operand 2) 
                  False (+ operand x)
                  (- operand x))))    

(print (. (test-cond 4) (block_until_ready)))


(macroexpand '(lcond [a b] (= a b) (+ a 1) (- b 2)))
(macroexpand '(if/j (jnp.less x 0) (jnp.add x 1) (jnp.subtract x 1)))
(macroexpand '(defn/j test-if [x]
                      (if/j (< x 3)
                            (* 3 (** x 2))
                            (* -4 x))))
(macroexpand-all '(defn/j test-cond [x]
                          (let [operand (jnp.array [0])]
                            (cond/j (< x 2) (+ operand 2) 
                                    (< x 4) (+ x 2)
                                    (- operand x)))))
(macroexpand '(cond/j (< x 3) (* 3 (** x 2))
                      (< x 5) (* -4 x)
                      (* 5 x)))

