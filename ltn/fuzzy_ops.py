from warnings import warn
import tensorflow as tf

"""
Element-wise fuzzy logic operators for tensorflow.
Supports traditional NumPy/Tensorflow broadcasting.

To use in LTN formulas (broadcasting w.r.t. ltn variables appearing in a formula), 
wrap the operator with `ltn.WrapperConnective` or `ltn.WrapperQuantifier`. 
"""

eps = 1e-4
not_zeros = lambda x: (1-eps)*x + eps
not_ones = lambda x: (1-eps)*x


class Not_Std:
    def __call__(self,x):
        return 1.-x


class Not_Godel:
    def __call__(self,x):
        return tf.cast(tf.equal(x,0),x.dtype)


class And_Min:
    def __call__(self,x,y):
        return tf.minimum(x,y)


class And_Prod:
    def __init__(self,stable=True):
        self.stable = stable
    
    def __call__(self,x,y,stable=None):
        stable = self.stable if stable is None else stable
        if stable:
            x, y = not_zeros(x), not_zeros(y)
        return tf.multiply(x,y)


class And_Luk:
    def __call__(self,x,y):
        return tf.maximum(x+y-1.,0.)


class Or_Max:
    def __call__(self,x,y):
        return tf.maximum(x,y)


class Or_ProbSum:
    def __init__(self,stable=True):
        self.stable = stable
    
    def __call__(self,x,y,stable=None):
        stable = self.stable if stable is None else stable
        if stable:
            x, y = not_ones(x), not_ones(y)
        return x + y - tf.multiply(x,y)


class Or_Luk:
    def __call__(self,x,y):
        return tf.minimum(x+y,1.)


class Implies_KleeneDienes:
    def __call__(self,x,y):
        return tf.maximum(1.-x,y)


class Implies_Godel:
    def __call__(self,x,y):
        return tf.where(tf.less_equal(x,y),tf.ones_like(x),y)

class UniNormImplication:
    def __init__(self, g):
        self.g = g
    
    def __call__(self, x, y):
        g = tf.constant(self.g, dtype=tf.float32)
        
        # Exemplo teórico: Comporta-se como uma implicação de Godel se x < g,
        # e como uma implicação de outra forma (por exemplo, Reichenbach) se x >= g.
        result = tf.where(
            tf.less(x, g),
            tf.where(tf.less_equal(x, y), tf.ones_like(x), y),  # Implicação de Godel
            1 - x + x * y  # Implicação de Reichenbach como exemplo alternativo
        )
        return result
    
class Implies_Reichenbach:
    def __init__(self,stable=True):
        self.stable = stable
    
    def __call__(self,x,y,stable=None):
        stable = self.stable if stable is None else stable
        if stable:
            x, y = not_zeros(x), not_ones(y)
        return 1.-x+tf.multiply(x,y)


class Implies_Goguen:
    def __init__(self,stable=True):
        self.stable = stable
    
    def __call__(self,x,y,stable=None):
        stable = self.stable if stable is None else stable
        if stable:
            x = not_zeros(x)
        return tf.where(tf.less_equal(x,y),tf.ones_like(x),tf.divide(y,x))


class Implies_Luk:
    def __call__(self,x,y):
        return tf.minimum(1.-x+y,1.)

#https://www.sciencedirect.com/science/article/pii/S0165011407002278        
class Implies_QL:
    def __init__(self, stable=True):
        self.stable = stable

    def strong_negation(self, x):
        if self.stable:
            x = not_zeros(x)
        return 1 - x

    def t_norm_conjunctive(self, x, y):
        if self.stable:
            x, y = not_zeros(x), not_zeros(y)
        # Usando mínimo como exemplo de t-norma conjuntiva
        return tf.minimum(x, y)

    def t_conorm_disjunctive(self, x, y):
        if self.stable:
            x, y = not_zeros(x), not_ones(y)
        # Usando máximo como exemplo de t-conorma disjuntiva
        return tf.maximum(x, y)

    def __call__(self, x, y, stable=None):
        stable = self.stable if stable is None else stable
        if stable:
            x, y = not_zeros(x), not_ones(y)
        N_x = self.strong_negation(x)
        U_x_y = self.t_norm_conjunctive(x, y)
        return self.t_conorm_disjunctive(N_x, U_x_y)


class Equiv:
    """Returns an operator that computes: And(Implies(x,y),Implies(y,x))"""
    def __init__(self, and_op, implies_op):
        self.and_op = and_op
        self.implies_op = implies_op
    
    def __call__(self, x, y):
        return self.and_op(self.implies_op(x,y), self.implies_op(y,x))


class Aggreg_Min:
    def __call__(self,xs,axis=None,keepdims=False):
        return tf.reduce_min(xs,axis=axis,keepdims=keepdims)


class Aggreg_Max:
    def __call__(self,xs,axis=None,keepdims=False):
        return tf.reduce_max(xs,axis=axis,keepdims=keepdims)


class Aggreg_Mean:
    def __call__(self,xs,axis=None,keepdims=False):
        return tf.reduce_mean(xs,axis=axis,keepdims=keepdims)


class Aggreg_pMean:
    def __init__(self,p=2,stable=True):
        self.p = p
        self.stable = stable
    
    def __call__(self,xs,axis=None,keepdims=False,p=None,stable=None):
        p = self.p if p is None else p 
        stable = self.stable if stable is None else stable
        if stable:
            xs = not_zeros(xs)
        return tf.pow(tf.reduce_mean(tf.pow(xs,p),axis=axis,keepdims=keepdims),1/p)


class Aggreg_pMeanError:
    def __init__(self,p=2,stable=True):
        self.p = p
        self.stable = stable
    
    def __call__(self,xs,axis=None,keepdims=False,p=None,stable=None):
        p = self.p if p is None else p
        stable = self.stable if stable is None else stable
        if stable:
            xs = not_ones(xs)
        return 1.-tf.pow(tf.reduce_mean(tf.pow(1.-xs,p),axis=axis,keepdims=keepdims),1/p)

class UniNormOperator:
    def __init__(self, g, t_norm, s_norm):
        self.g = g
        # Assume que t_norm e s_norm são instâncias das classes And_Prod e Or_ProbSum
        self.t_norm = t_norm
        self.s_norm = s_norm

    def __call__(self, x, y):
        g = tf.constant(self.g, dtype=tf.float32)
        condition_t_norm = tf.logical_and(x < g, y < g)
        condition_s_norm = tf.logical_and(x > g, y > g)

        # Ajuste: removendo a passagem explícita do argumento 'stable'
        result_t_norm = tf.where(condition_t_norm, tf.multiply(x / g, y / g), x)
        result_s_norm = tf.where(condition_s_norm, x + y - tf.multiply((x - g) / (1 - g), (y - g) / (1 - g)), x)

        result_else = tf.where(g > 0.5, tf.minimum(x, y), tf.maximum(x, y))
        result = tf.where(condition_t_norm, result_t_norm, 
                          tf.where(condition_s_norm, result_s_norm, result_else))
        return result

class Aggreg_Prod:
    def __call__(self,xs,axis=None,keepdims=False):
        return tf.reduce_prod(xs,axis=axis,keepdims=keepdims)
    

class Aggreg_LogProd:
    def __init__(self,stable=True):
        warn("`Aggreg_LogProd` outputs values out of the truth value range [0,1]. "
             "Its usage with other connectives could be compromised."
             "Use it carefully.", UserWarning)
        self.stable = stable

    def __call__(self,xs,stable=None,axis=None, keepdims=False):
        stable = self.stable if stable is None else stable
        if stable:
            xs=not_zeros(xs)
        return tf.reduce_sum(tf.math.log(xs),axis=axis,keepdims=keepdims)
    
class Xor_Uninorm:
    """
    Fuzzy XOR operator based on uninorm concepts.
    This operator attempts to capture the essence of XOR in a fuzzy logic context,
    emphasizing the difference between the inputs.
    """
    def __init__(self, g=0.5):
        # g is a parameter that could adjust the behavior of the fuzzy XOR,
        # but will be kept fixed for simplicity in this example.
        self.g = g

    def __call__(self, x, y):
        # Calculate the "fuzzy divergence" as a measure for XOR logic.
        divergence = tf.abs(x - y)
        
        # Fuzzy XOR result: Higher divergence indicates a result closer to true (1),
        # while lower divergence indicates false (0).
        # The result is adjusted by g to determine the transition behavior.
        xor_result = tf.where(divergence <= self.g, divergence / self.g, 1.0 - (divergence - self.g) / (1.0 - self.g))
        
        return xor_result
class Implies_Reversed_Uninorm:
    """
    Fuzzy reversed implication operator based on uninorms.
    This operator models the logic of reversed implication (y implies x) using uninorm concepts.
    """
    def __init__(self, g, t_norm, s_norm):
        self.g = g  # Control parameter for the uninorm
        # t_norm and s_norm are instances of classes representing t-norm and s-norm operations
        self.t_norm = t_norm
        self.s_norm = s_norm

    def __call__(self, x, y):
        g = tf.constant(self.g, dtype=tf.float32)
        
        # Applying reversed implication logic using uninorms
        # The approach here is conceptual and aims to illustrate the use of uninorms for reversed implication
        # Calculate the negation of y using the strong negation concept
        N_y = 1 - y
        
        # Use the t_norm for cases where y <= g, implying a stronger condition for reversal
        result_t_norm = self.t_norm(N_y, x)
        
        # Use the s_norm for cases where y > g, allowing a more permissive condition for reversal
        result_s_norm = self.s_norm(N_y, x)
        
        # Determine which result to use based on the value of y relative to g
        result = tf.where(y <= g, result_t_norm, result_s_norm)
        
        return result
    
    
Aggreg_SumLog = Aggreg_LogProd