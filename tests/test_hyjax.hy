(import jax
        jax [numpy :as jnp
             grad
             random])

(require hyjax [defn/j mapv if/l cond/l while/l fori/l while/lp fori/lp
                scan/l scan/lp case/l ascan/l mapl mapt
                jvp/l vjp/l with-keys])

;;; defn/j — basic jit

(defn/j selu [x [alpha 1.67] [lmbda 1.05]]
  (* lmbda (jnp.where (> x 0) x (- (* alpha (jnp.exp x)) alpha))))

(defn test-defn-j-is-jitted []
  "defn/j should wrap the function with jax.jit."
  ;; jitted callables are instances of jax's PjitFunction / CompiledFunction.
  ;; The robust check is that lowering works without tracing errors.
  (let [y (selu (jnp.array 1.0))]
    (assert (= y.shape #()))
    (assert (jnp.allclose y 1.05))))

(defn test-defn-j-with-defaults []
  (let [x (jnp.array [-1.0 0.0 2.0])
        y (selu x)]
    (assert (= y.shape #(3)))))

;;; mapv — vmap sugar

(defn/j square [x] (* x x))

(defn test-mapv-squares-vector []
  (let [v (jnp.arange 5.0)
        out (mapv square v)]
    (assert (jnp.allclose out (jnp.array [0.0 1.0 4.0 9.0 16.0])))))

(defn test-mapv-matrix-vector []
  (let [mat (jnp.eye 3)]
    (defn/j apply-mat [v] (jnp.dot mat v))
    (let [batch (jnp.arange 6.0 :dtype jnp.float32)
          batch (jnp.reshape batch #(2 3))
          out (mapv apply-mat batch)]
      (assert (jnp.allclose out batch)))))

;;; if/l — compiles to lax.cond

(defn/j branch [x]
  (if/l (< x 3)
        (* 3 (** x 2))
        (* -4 x)))

(defn test-if-j-true-branch []
  (assert (= (int (branch (jnp.array 2))) 12)))

(defn test-if-j-false-branch []
  (assert (= (int (branch (jnp.array 5))) -20)))

;;; nested if/l

(defn/j nested [x]
  (if/l (< x 3) (* 3 (** x 2))
        (if/l (< x 5) (* -4 x)
              (* 5 x))))

(defn test-nested-if-j []
  (assert (= (int (nested (jnp.array 2))) 12))
  (assert (= (int (nested (jnp.array 4))) -16))
  (assert (= (int (nested (jnp.array 10))) 50)))

;;; cond/l — multi-branch lax.cond

(defn/j piecewise [x]
  (cond/l
    (< x 3) (* 3 (** x 2))
    (< x 5) (* -4 x)
    (* 5 x)))

(defn test-cond-j-first-branch []
  (assert (= (int (piecewise (jnp.array 2))) 12)))

(defn test-cond-j-middle-branch []
  (assert (= (int (piecewise (jnp.array 4))) -16)))

(defn test-cond-j-fallthrough []
  (assert (= (int (piecewise (jnp.array 10))) 50)))

;;; while/l — pure expression, carry names explicit, inits from scope

(defn/j count-to [n]
  (setv v (jnp.int32 0))
  (while/l [v] (< v n) (+ v 1)))

(defn test-while-l-counts []
  (assert (= (int (count-to (jnp.array 7))) 7))
  (assert (= (int (count-to (jnp.array 0))) 0)))

(defn/j collatz-steps [start]
  ;; number of Collatz steps from `start` until we reach 1.
  (setv v     (jnp.int32 start))
  (setv steps (jnp.int32 0))
  (get (while/l [v steps]
                (> v 1)
                #((jnp.where (= (% v 2) 0)
                             (// v 2)
                             (+ (* 3 v) 1))
                  (+ steps 1)))
       1))

(defn test-while-l-multi-state []
  ;; 27 -> 1 takes 111 Collatz steps.
  (assert (= (int (collatz-steps (jnp.array 27))) 111)))

;;; fori/l — pure expression, carry names explicit, inits from scope

(defn/j sum-range [n]
  (setv acc (jnp.int32 0))
  (fori/l [acc] [i (range n)] (+ acc i)))

(defn test-fori-l-sum []
  ;; sum of 0..9 = 45
  (assert (= (int (sum-range 10)) 45))
  (assert (= (int (sum-range 0)) 0)))

(defn/j factorial [n]
  (setv acc (jnp.int32 1))
  (fori/l [acc] [i (range 1 (+ n 1))] (* acc i)))

(defn test-fori-l-factorial []
  (assert (= (int (factorial 5)) 120))
  (assert (= (int (factorial 1)) 1)))

(defn/j fib [n]
  ;; Fibonacci: carry (a, b); new state is #(b (+ a b)).
  (setv a (jnp.int32 0))
  (setv b (jnp.int32 1))
  (get (fori/l [a b] [i (range n)] #(b (+ a b))) 0))

(defn test-fori-l-multi-state []
  (assert (= (int (fib 0)) 0))
  (assert (= (int (fib 1)) 1))
  (assert (= (int (fib 10)) 55)))

;;; while/lp / fori/lp — explicit-binding forms

(defn/j count-to-explicit [n]
  (while/lp [v (jnp.int32 0)]
          (< v n)
          (+ v 1)))

(defn test-while-lp-counts []
  (assert (= (int (count-to-explicit (jnp.array 4))) 4)))

(defn/j collatz-explicit [start]
  ;; Same Collatz step counter, but with the carry written by hand.
  (let [result
        (while/lp [[v steps] #((jnp.int32 start) (jnp.int32 0))]
                (> v 1)
                #((jnp.where (= (% v 2) 0)
                             (// v 2)
                             (+ (* 3 v) 1))
                  (+ steps 1)))]
    (get result 1)))

(defn test-while-lp-multi-state []
  (assert (= (int (collatz-explicit (jnp.array 27))) 111)))

(defn/j sum-range-explicit [n]
  (fori/lp [acc (jnp.int32 0)] [i 0 n]
         (+ acc i)))

(defn test-fori-lp-sum []
  (assert (= (int (sum-range-explicit 10)) 45)))

(defn/j fib-explicit [n]
  (let [result
        (fori/lp [[a b] #((jnp.int32 0) (jnp.int32 1))] [i 0 n]
               #(b (+ a b)))]
    (get result 0)))

(defn test-fori-lp-multi-state []
  (assert (= (int (fib-explicit 10)) 55)))

;;; scan/l — lax.scan wrapper

(defn/j cumsum-scan [xs]
  (setv acc (jnp.int32 0))
  (get (scan/l [acc] [x xs] #((+ acc x) (+ acc x))) 1))

(defn test-scan-l-cumsum []
  (let [xs (jnp.array [1 2 3 4 5])
        out (cumsum-scan xs)]
    (assert (jnp.array_equal out (jnp.array [1 3 6 10 15])))))

(defn/j scan-fib [dummy]
  ;; stacked Fibonacci sequence via multi-carry scan. `dummy` sets
  ;; the scan length via its leading axis.
  (setv a (jnp.int32 0))
  (setv b (jnp.int32 1))
  (get (scan/l [a b] [_ dummy] #(#(b (+ a b)) a)) 1))

(defn test-scan-l-multi-carry []
  (assert (jnp.array_equal (scan-fib (jnp.zeros 8))
                           (jnp.array [0 1 1 2 3 5 8 13]))))

(defn/j scan-lp-sum [xs]
  (get (scan/lp [acc (jnp.int32 0)] [x xs] #((+ acc x) (+ acc x))) 0))

(defn test-scan-lp []
  (assert (= (int (scan-lp-sum (jnp.array [1 2 3 4 5]))) 15)))

;;; case/l — lax.switch wrapper

(defn/j dispatch [idx x]
  (case/l idx
          (* x 2)
          (+ x 10)
          (- x)))

(defn test-case-l []
  (assert (= (int (dispatch 0 (jnp.array 5))) 10))
  (assert (= (int (dispatch 1 (jnp.array 5))) 15))
  (assert (= (int (dispatch 2 (jnp.array 5))) -5)))

;;; ascan/l — lax.associative_scan wrapper

(defn/j prefix-sums [xs]
  (ascan/l (fn [a b] (+ a b)) xs))

(defn test-ascan-l []
  (assert (jnp.array_equal (prefix-sums (jnp.array [1 2 3 4]))
                           (jnp.array [1 3 6 10]))))

;;; mapl — lax.map wrapper (sequential)

(defn/j square-each [xs]
  (mapl (fn [x] (* x x)) xs))

(defn test-mapl []
  (assert (jnp.array_equal (square-each (jnp.array [1 2 3 4]))
                           (jnp.array [1 4 9 16]))))

;;; mapt — jax.tree.map wrapper

(defn test-mapt-single-tree []
  (let [tree {"a" (jnp.array [1 2]) "b" (jnp.array [3 4])}
        out (mapt (fn [x] (* x 10)) tree)]
    (assert (jnp.array_equal (get out "a") (jnp.array [10 20])))
    (assert (jnp.array_equal (get out "b") (jnp.array [30 40])))))

(defn test-mapt-multi-tree []
  ;; gradient-descent-style update over parameter pytree
  (let [params {"w" (jnp.array 1.0) "b" (jnp.array 2.0)}
        grads  {"w" (jnp.array 0.1) "b" (jnp.array 0.2)}
        lr 0.5
        updated (mapt (fn [p g] (- p (* lr g))) params grads)]
    (assert (jnp.allclose (get updated "w") 0.95))
    (assert (jnp.allclose (get updated "b") 1.9))))

;;; jvp/l and vjp/l — autodiff pairs

(defn test-jvp-l []
  ;; f(x) = x^2; f'(x) = 2x; at x=3, dx=1 -> dy = 6
  (let [[y dy] (jvp/l [(jnp.float32 3.0)] [(jnp.float32 1.0)]
                      (fn [a] (* a a)))]
    (assert (jnp.allclose y 9.0))
    (assert (jnp.allclose dy 6.0))))

(defn test-jvp-l-multi-primal []
  ;; f(a, b) = a*b; df = b*da + a*db; at (3,4), (1,1) -> 7
  (let [[y dy] (jvp/l [(jnp.float32 3.0) (jnp.float32 4.0)]
                      [(jnp.float32 1.0) (jnp.float32 1.0)]
                      (fn [a b] (* a b)))]
    (assert (jnp.allclose y 12.0))
    (assert (jnp.allclose dy 7.0))))

(defn test-vjp-l []
  ;; f(x) = x^3; f'(x) = 3x^2; at x=2 -> grad = 12
  (let [[y vjp-fn] (vjp/l [(jnp.float32 2.0)] (fn [a] (* a (* a a))))
        [dx] (vjp-fn (jnp.float32 1.0))]
    (assert (jnp.allclose y 8.0))
    (assert (jnp.allclose dx 12.0))))

(defn test-vjp-l-multi-primal []
  ;; f(a, b) = a^2 + b^2; gradients are (2a, 2b); at (3,4) -> (6, 8)
  (let [[y vjp-fn] (vjp/l [(jnp.float32 3.0) (jnp.float32 4.0)]
                          (fn [a b] (+ (* a a) (* b b))))
        [da db] (vjp-fn (jnp.float32 1.0))]
    (assert (jnp.allclose y 25.0))
    (assert (jnp.allclose da 6.0))
    (assert (jnp.allclose db 8.0))))

;;; with-keys — PRNG split sugar

(defn test-with-keys []
  (import jax [random])
  (setv key (random.PRNGKey 0))
  (setv orig-key key)
  (with-keys key [k1 k2 k3]
    (setv x (random.normal k1 #(4)))
    (setv y (random.normal k2 #(4)))
    (setv z (random.normal k3 #(4))))
  ;; key should have been consumed/advanced
  (assert (not (jnp.array_equal key orig-key)))
  ;; sub-keys must all be distinct
  (assert (not (jnp.array_equal k1 k2)))
  (assert (not (jnp.array_equal k2 k3)))
  (assert (not (jnp.array_equal k1 k3)))
  ;; generated samples should be different draws
  (assert (not (jnp.allclose x y))))

;;; defn/j composes with jax.grad

(defn/j sum-logistic [x]
  (jnp.sum (/ 1.0 (+ 1.0 (jnp.exp (- x))))))

(defn test-defn-j-is-differentiable []
  (let [x (jnp.arange 3.0)
        g ((grad sum-logistic) x)
        expected (jnp.array [0.25 0.19661197 0.10499357])]
    (assert (jnp.allclose g expected :atol 1e-6))))
