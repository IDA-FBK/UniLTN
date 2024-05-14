import ltn
import tensorflow as tf


@tf.function
def axioms_binaryc(data, labels, axioms_info):
    """
    Construct and evaluate axioms for binary classification task.

    Parameters:
    - data (tf.Tensor): Input data features.
    - labels (tf.Tensor): Binary class labels.
    - axioms_info (dict): Dictionary containing axioms info such as logical operations and predicates.

    Returns:
    - tf.Tensor: Satisfaction level of the constructed axioms.
    """
    Not = axioms_info["not"]
    Forall = axioms_info["forall"]
    formula_aggregator = axioms_info["fagg"]
    A = axioms_info["A"]

    x_A = ltn.Variable("x_A", data[labels])
    x_not_A = ltn.Variable("x_not_A", data[tf.logical_not(labels)])

    axioms = [
        Forall(x_A, A(x_A)),
        Forall(x_not_A, Not(A(x_not_A)))
    ]

    sat_level = formula_aggregator(axioms).tensor

    return sat_level


@tf.function
def axioms_clustering(axioms_info, p_exists):
    """
    Construct and evaluate axioms for clustering tasks.

    Parameters:
    - axioms_info (dict): Dictionary containing logical operations, predicates, and parameters.
    - p_exists (int): p parameter for the exist quantifier

    Returns:
    - tf.Tensor: Satisfaction level of the constructed axioms.
    """
    Not = axioms_info["not"]
    And = axioms_info["and"]
    Forall = axioms_info["forall"]
    Exists = axioms_info["exists"]
    Equiv = axioms_info["equiv"]
    formula_aggregator = axioms_info["fagg"]

    x = axioms_info["x"]
    y = axioms_info["y"]
    cluster = axioms_info["cluster"]
    C = axioms_info["C"]

    close_thr = axioms_info["close_thr"]
    distant_thr = axioms_info["distant_thr"]
    eucl_dist = axioms_info["eucl_dist"]
    is_greater_than = axioms_info["is_greater_than"]

    axioms = [
        Forall(x, Exists(cluster, C([x, cluster]), p=p_exists)),
        Forall(cluster, Exists(x, C([x, cluster]), p=p_exists)),
        Forall([cluster, x, y], Equiv(C([x, cluster]), C([y, cluster])),
               mask=is_greater_than([close_thr, eucl_dist([x, y])])),
        Forall([cluster, x, y], Not(And(C([x, cluster]), C([y, cluster]))),
               mask=is_greater_than([eucl_dist([x, y]), distant_thr]))
    ]
    sat_level = formula_aggregator(axioms).tensor

    return sat_level


@tf.function
def axioms_mnist_singled(images_x, images_y, labels_z, axioms_info, p_schedule):
    """
    Construct and evaluate axioms for mnist single digit addition.

    Parameters:
    - images_x (tf.Tensor): Input images for the first operand.
    - images_y (tf.Tensor): Input images for the second operand.
    - labels_z (tf.Tensor): Target labels.
    - axioms_info (dict): Dictionary containing axioms info such as logical operations and predicates.
    - p_schedule (tf.Tensor): schedule parameter.

    Returns:
    - tf.Tensor: Satisfaction level of the constructed axioms.
    """
    And = axioms_info["and"]
    Forall = axioms_info["forall"]
    Exists = axioms_info["exists"]

    Digit = axioms_info["Digit"]
    add = axioms_info["add"]
    equals = axioms_info["equals"]
    d1, d2 = axioms_info["d1"], axioms_info["d2"]

    images_x = ltn.Variable("x", images_x)
    images_y = ltn.Variable("y", images_y)
    labels_z = ltn.Variable("z", labels_z)
    axiom = Forall(
            ltn.diag(images_x, images_y, labels_z),
            Exists(
                (d1, d2),
                And(Digit([images_x, d1]), Digit([images_y, d2])),
                mask=equals([add([d1, d2]), labels_z]),
                p=p_schedule
            ),
            p=2
        )
    sat = axiom.tensor
    return sat


@tf.function
def axioms_mnist_multid(images_x1, images_x2, images_y1, images_y2, labels_z, axioms_info, p_schedule):
    """
    Construct and evaluate axioms for mnist multi digit addition.

    Parameters:
    - images_x (tf.Tensor): Input images for the first operand.
    - images_y (tf.Tensor): Input images for the second operand.
    - labels_z (tf.Tensor): Target labels.
    - axioms_info (dict): Dictionary containing axioms info such as logical operations and predicates.
    - p_schedule (tf.Tensor): schedule parameter.

    Returns:
    - tf.Tensor: Satisfaction level of the constructed axioms.
    """
    images_x1 = ltn.Variable("x1", images_x1)
    images_x2 = ltn.Variable("x2", images_x2)
    images_y1 = ltn.Variable("y1", images_y1)
    images_y2 = ltn.Variable("y2", images_y2)
    labels_z = ltn.Variable("z", labels_z)

    And = axioms_info["and"]
    Forall = axioms_info["forall"]
    Exists = axioms_info["exists"]
    Digit = axioms_info["Digit"]

    d1 = axioms_info["d1"]
    d2 = axioms_info["d2"]
    d3 = axioms_info["d3"]
    d4 = axioms_info["d4"]

    add = axioms_info["add"]
    equals = axioms_info["equals"]
    two_digit_number = axioms_info["two_digit_number"]

    axiom = Forall(
            ltn.diag(images_x1, images_x2, images_y1, images_y2, labels_z),
            Exists(
                (d1, d2, d3, d4),
                And(
                    And(Digit([images_x1, d1]), Digit([images_x2,d2])),
                    And(Digit([images_y1, d3]), Digit([images_y2, d4]))
                ),
                mask=equals([labels_z, add([two_digit_number([d1, d2]), two_digit_number([d3, d4])])]),
                p=p_schedule
            ),
            p=2
        )
    sat = axiom.tensor
    return sat


@tf.function
def axioms_multiclass(features, labels, axioms_info):
    """
    Construct and evaluate axioms for multiclass classification.

    Parameters:
    - features (tf.Tensor): Input features.
    - labels (tf.Tensor): Target labels.
    - axioms_info (dict): Dictionary containing axioms info such as logical operations and predicates.

    Returns:
    - tf.Tensor: Satisfaction level of the constructed axioms.
    """

    class_A = axioms_info["class_A"]
    class_B = axioms_info["class_B"]
    class_C = axioms_info["class_C"]

    p = axioms_info["p"]

    Forall = axioms_info["forall"]
    formula_aggregator = axioms_info["fagg"]
    training = axioms_info["training"]

    x_A = ltn.Variable("x_A", features[labels == 0])
    x_B = ltn.Variable("x_B", features[labels == 1])
    x_C = ltn.Variable("x_C", features[labels == 2])

    axioms = [
        Forall(x_A,p([x_A,class_A],training=training)),
        Forall(x_B,p([x_B,class_B],training=training)),
        Forall(x_C,p([x_C,class_C],training=training))
    ]
    sat_level = formula_aggregator(axioms).tensor
    return sat_level


@tf.function()
def sat_phi1(features, axioms_info):
    """
    Calculate the satisfaction level of the phi2 formula:

        `forall x (p(x,blue)->~p(x,orange))` (every blue crab cannot be orange and vice-versa)

    we expect to be true.

    Parameters:
    - features (tf.Tensor): Input features.
    - axioms_info (dict): Dictionary containing axioms info such as logical operations and predicates.

    Returns:
    - tf.Tensor: Satisfaction level of the phi3 formula.
    """
    class_blue = axioms_info["class_blue"]
    class_orange = axioms_info["class_orange"]

    Not = axioms_info["not"]
    Forall = axioms_info["forall"]
    Implies = axioms_info["implies"]
    x = ltn.Variable("x", features)
    p = axioms_info["p"]

    phi1 = Forall(x, Implies(p([x, class_blue]), Not(p([x, class_orange]))), p=5)
    return phi1.tensor


@tf.function()
def sat_phi2(features, axioms_info):
    """
    Calculate the satisfaction level of the phi2 formula:

        `forall x (p(x,blue)->p(x,orange))` (every blue crab is also orange)

    we expect to be false.

    Parameters:
    - features (tf.Tensor): Input features.
    - axioms_info (dict): Dictionary containing axioms info such as logical operations and predicates.

    Returns:
    - tf.Tensor: Satisfaction level of the phi3 formula.
    """
    class_blue = axioms_info["class_blue"]
    class_orange = axioms_info["class_orange"]

    Forall = axioms_info["forall"]
    Implies = axioms_info["implies"]
    x = ltn.Variable("x",features)
    p = axioms_info["p"]

    phi2 = Forall(x, Implies(p([x, class_blue]), p([x, class_orange])), p=5)
    return phi2.tensor


@tf.function()
def sat_phi3(features, axioms_info):
    """
    Calculate the satisfaction level of the phi3 formula:

        `forall x (p(x,blue)->p(x,male))` ---- (every blue crab is male)

    we expect to be false.

    Parameters:
    - features (tf.Tensor): Input features.
    - axioms_info (dict): Dictionary containing axioms info such as logical operations and predicates.

    Returns:
    - tf.Tensor: Satisfaction level of the phi3 formula.
    """
    class_male = axioms_info["class_male"]
    class_blue = axioms_info["class_blue"]

    Forall = axioms_info["forall"]
    Implies = axioms_info["implies"]
    x = ltn.Variable("x", features)
    p = axioms_info["p"]

    phi3 = Forall(x, Implies(p([x, class_blue]), p([x, class_male])), p=5)
    return phi3.tensor


@tf.function
def axioms_multilabel(features, labels_sex, labels_color, axioms_info):
    """
    Construct and evaluate axioms for multilabel classification.

    Parameters:
    - features (tf.Tensor): Input features.
    - labels_sex (tf.Tensor): Target labels for sex.
    - labels_color (tf.Tensor): Target labels for color.
    - axioms_info (dict): Dictionary containing axioms info such as logical operations and predicates.

    Returns:
    - tf.Tensor: Satisfaction level of the constructed axioms.
    """
    class_male = axioms_info["class_male"]
    class_female = axioms_info["class_female"]
    class_blue = axioms_info["class_blue"]
    class_orange = axioms_info["class_orange"]

    Forall = axioms_info["forall"]
    And = axioms_info["and"]
    Not = axioms_info["not"]
    p = axioms_info["p"]
    formula_aggregator = axioms_info["fagg"]

    x = ltn.Variable("x",features)
    x_blue = ltn.Variable("x_blue", features[labels_color == "B"])
    x_orange = ltn.Variable("x_orange", features[labels_color == "O"])
    x_male = ltn.Variable("x_blue", features[labels_sex == "M"])
    x_female = ltn.Variable("x_blue", features[labels_sex == "F"])

    # Axioms
    #   forall x_blue: C(x_blue,blue)
    #   forall x_orange: C(x_orange,orange)
    #   forall x_male: C(x_male,male)
    #   forall x_female: C(x_female,female)
    #   forall x: ~(C(x,male) & C(x,female))
    #   forall x: ~(C(x,blue) & C(x,orange))

    axioms = [
        Forall(x_blue, p([x_blue, class_blue])),
        Forall(x_orange, p([x_orange, class_orange])),
        Forall(x_male, p([x_male, class_male])),
        Forall(x_female, p([x_female, class_female])),
        Forall(x, Not(And(p([x, class_blue]), p([x, class_orange])))),
        Forall(x, Not(And(p([x, class_male]), p([x, class_female]))))
    ]
    sat_level = formula_aggregator(axioms).tensor
    return sat_level


@tf.function
def axioms_parent_ancestor(axioms_info):
    """
    Construct and evaluate axioms for parent-ancestor relationships.

    Parameters:
    - axioms_info (dict): Dictionary containing axioms info such as logical operations and predicates.

    Returns:
    - tf.Tensor: Satisfaction level of the constructed axioms.
    """

    g_e = axioms_info["g_e"]

    Parent = axioms_info["parent_predicate"]
    Ancestor = axioms_info["ancestor_predicate"]

    parents = axioms_info["parents"]
    not_parents = axioms_info["not_parents"]

    Not = axioms_info["not"]
    And = axioms_info["and"]
    Or = axioms_info["or"]
    Forall = axioms_info["forall"]
    Exists = axioms_info["exists"]
    Implies = axioms_info["implies"]
    formula_aggregator = axioms_info["fagg"]

    # Variables created in the training loop, so tf.GradientTape
    # keeps track of the connection with the trainable constants.
    a = ltn.Variable.from_constants("a", list(g_e.values()))
    b = ltn.Variable.from_constants("b", list(g_e.values()))
    c = ltn.Variable.from_constants("c", list(g_e.values()))

    ## Complete knowledge about parent relationships.
    ## The ancestor relationships are to be learned with these additional rules.
    axioms = [
        # forall pairs of individuals in the parent relationships: Parent(ancestor,child)
        Parent([g_e[a],g_e[c]])
        for a,c in parents
    ] + \
    [
        # forall pairs of individuals not in the parent relationships: Not(Parent([n_parent,n_child])))
        Not(Parent([g_e[a],g_e[c]]))
        for a,c in not_parents
    ] + \
    [
        # if a is parent of b, then a is ancestor of b
        Forall((a,b), Implies(Parent([a,b]),Ancestor([a,b]))),
        # parent is anti reflexive
        Forall(a, Not(Parent([a,a]))),
        # ancestor is anti reflexive
        Forall(a, Not(Ancestor([a,a]))),
        # parent is anti symmetric
        Forall((a,b), Implies(Parent([a,b]),Not(Parent([b,a])))),
        # if a is parent of an ancestor of c, a is an ancestor of c too
        Forall(
            (a,b,c),
            Implies(And(Parent([a,b]),Ancestor([b,c])), Ancestor([a,c])),
            p=6
        ),
        # if a is an ancestor of b, a is a parent of b OR a parent of an ancestor of b
        Forall(
            (a,b),
            Implies(Ancestor([a,b]),
                    Or(Parent([a,b]),
                       Exists(c, And(Ancestor([a,c]),Parent([c,b])),p=6)
                      )
                   )
        )
    ]
    # computing sat_level
    sat_level = formula_aggregator(axioms).tensor

    return sat_level


@tf.function
def axioms_propositional_variables(axioms_info):
    """
    Construct and evaluate axioms involving propositional variables.

    Parameters:
    - axioms_info (dict): Dictionary containing logical operations, predicates, variables, and propositions.

    Returns:
    - tf.Tensor: Satisfaction level of the constructed axioms.
    """
    a = axioms_info["a"]
    b = axioms_info["b"]
    c = axioms_info["c"]
    d = axioms_info["d"]
    e = axioms_info["e"]
    x = axioms_info["x"]
    P = axioms_info["P"]

    Not = axioms_info["not"]
    And = axioms_info["and"]
    Forall = axioms_info["forall"]
    Exists = axioms_info["exists"]
    Implies = axioms_info["implies"]
    formula_aggregator = axioms_info["fagg"]

    axioms = [
        # [ (A and B and (forall x: P(x))) -> Not C ] and C
        And(
            Implies(And(And(a,b),Forall(x,P(x))),
                    Not(c)),
            c
        ),
        # w1 -> (forall x: P(x))
        Implies(d, Forall(x,P(x))),
        # w2 -> (Exists x: P(x))
        Implies(e, Exists(x,P(x)))
    ]

    sat_level = formula_aggregator(axioms).tensor
    return sat_level


@tf.function
def axioms_regression(x_data, y_data, axioms_info):
    """
    Construct and evaluate axioms for regression tasks.

    Parameters:
    - x_data (tf.Tensor): Input data features.
    - y_data (tf.Tensor): Target values.
    - axioms_info (dict): Dictionary containing logical operations and predicates.

    Returns:
    - tf.Tensor: Satisfaction level of the constructed axioms.
    """

    x = ltn.Variable("x", x_data)
    y = ltn.Variable("y", y_data)

    f = axioms_info["f"]
    Forall = axioms_info["forall"]
    eq = axioms_info["eq"]

    return Forall(ltn.diag(x, y), eq([f(x), y])).tensor


# defining the theory
@tf.function
def axioms_smokes_friends_cancer(axioms_info, p_exists):
    """
    NOTE: we update the embeddings at each step
        -> we should re-compute the variables.
    """
    g = axioms_info["g"]
    g1 = axioms_info["g1"]
    g2 = axioms_info["g2"]
    # predicates
    Smokes = axioms_info["Smokes"]
    Friends = axioms_info["Friends"]
    Cancer = axioms_info["Cancer"]

    smokes = axioms_info["smokes"]
    friends = axioms_info["friends"]
    cancer = axioms_info["cancer"]

    p = ltn.Variable.from_constants("p", list(g.values()))
    q = ltn.Variable.from_constants("q", list(g.values()))

    And = axioms_info["and"]
    Not = axioms_info["not"]
    Forall = axioms_info["forall"]
    Exists = axioms_info["exists"]
    Implies = axioms_info["implies"]
    formula_aggregator = axioms_info["fagg"]
    axioms = []

    # Friends: knowledge incomplete in that
    #     Friend(x,y) with x<y may be known
    #     but Friend(y,x) may not be known
    axioms.append(formula_aggregator(
            [Friends([g[x],g[y]]) for (x,y) in friends]))
    axioms.append(formula_aggregator(
            [Not(Friends([g[x],g[y]])) for x in g1 for y in g1 if (x,y) not in friends and x<y ]+\
            [Not(Friends([g[x],g[y]])) for x in g2 for y in g2 if (x,y) not in friends and x<y ]))
    # Smokes: knowledge complete
    axioms.append(formula_aggregator(
            [Smokes(g[x]) for x in smokes]))
    axioms.append(formula_aggregator(
            [Not(Smokes(g[x])) for x in g if x not in smokes]))
    # Cancer: knowledge complete in g1 only
    axioms.append(formula_aggregator(
            [Cancer(g[x]) for x in cancer]))
    axioms.append(formula_aggregator(
            [Not(Cancer(g[x])) for x in g1 if x not in cancer]))
    # friendship is anti-reflexive
    axioms.append(Forall(p,Not(Friends([p,p])),p=5))
    # friendship is symmetric
    axioms.append(Forall((p,q),Implies(Friends([p,q]),Friends([q,p])),p=5))
    # everyone has a friend
    axioms.append(Forall(p,Exists(q,Friends([p,q]),p=p_exists)))
    # smoking propagates among friends
    axioms.append(Forall((p,q),Implies(And(Friends([p,q]),Smokes(p)),Smokes(q))))
    # smoking causes cancer + not smoking causes not cancer
    axioms.append(Forall(p,Implies(Smokes(p),Cancer(p))))
    axioms.append(Forall(p,Implies(Not(Smokes(p)),Not(Cancer(p)))))
    # computing sat_level
    sat_level = formula_aggregator(axioms).tensor
    return sat_level
