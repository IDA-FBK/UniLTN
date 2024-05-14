import math

import ltn
import numpy as np
import pandas as pd
import tensorflow as tf

from experiments.plots import plots_smokes_friend_cancer_facts, visualize_embeddings


def evalute_additional_queries_parent_ancestors(task_info, log_path):
    """
    Evaluate additional logical queries related to parent-ancestor relationships and save results to a file.

    Parameters:
    - task_info (dict): Dictionary containing logical operations and predicates.
    - log_path (str): Path to the log file.

    Returns:
    None
    """

    g_e = task_info["g_e"]
    And = task_info["and"]
    Not = task_info["not"]
    Implies = task_info["implies"]
    Forall = task_info["forall"]
    Parent = task_info["parent_predicate"]
    Ancestor = task_info["ancestor_predicate"]


    a = ltn.Variable.from_constants("a", list(g_e.values()))
    b = ltn.Variable.from_constants("b", list(g_e.values()))
    c = ltn.Variable.from_constants("c", list(g_e.values()))
    with open(log_path + "additional_queries.txt", 'w') as f:

        # q1: forall a,b,c: (Ancestor(a,b) & Parent(b,c)) -> Ancestor (a, c)
        q1_res = Forall((a, b, c), Implies(And(Ancestor([a, b]), Parent([b, c])), Ancestor([a, c]))).tensor.numpy()
        q1_str = "q1: forall a,b,c: (Ancestor(a,b) & Parent(b,c)) -> Ancestor (a, c) = %.3f\n\n" % q1_res
        print(q1_str)
        f.write(q1_str)

        # q2: forall a,b: Ancestor(a,b) -> ~Ancestor(b,a)
        q2_res = Forall((a, b), Implies(Ancestor([a, b]), Not(Ancestor([b, a])))).tensor.numpy()
        q2_str = "q2: forall a,b: Ancestor(a,b) -> ~Ancestor(b,a) = %.3f\n\n" % q2_res
        print(q2_str)
        f.write(q2_str)

        # q3: forall a,b,c: (Parent(a,b) & Parent(b,c)) -> Ancestor(a,c)
        q3_res = Forall((a, b, c), Implies(And(Parent([a, b]), Parent([b, c])), Ancestor([a, c]))).tensor.numpy()
        q3_str = "q3: forall a,b,c: (Parent(a,b) & Parent(b,c)) -> Ancestor(a,c) = %.3f\n\n" % q3_res
        print(q3_str)
        f.write(q3_str)

        # q4: forall a,b,c: (Ancestor(a,b) & Ancestor(b,c)) -> Ancestor(a,c)
        q4_res = Forall((a, b, c), Implies(And(Parent([a, b]), Parent([b, c])), Ancestor([a, c]))).tensor.numpy()
        q4_str = "q4: forall a,b,c: (Ancestor(a,b) & Ancestor(b,c)) -> Ancestor(a,c) = %.3f\n\n" % q4_res
        print(q4_str)
        f.write(q4_str)


def evaluate_smokes_friend_cancer(evaluation_info, path_to_results):
    """
    Evaluate and visualize facts and queries related to smoking, friendship, and cancer.

    Parameters:
    - evaluation_info (dict): Dictionary containing information needed for evaluation.
    - path_to_results (str): Path to save the results.

    Returns:
    None
    """

    np.set_printoptions(suppress=True)

    pd.options.display.max_rows = 999
    pd.options.display.max_columns = 999
    pd.set_option('display.width', 1000)
    pd.options.display.float_format = '{:,.2f}'.format

    #pd.set_option('precision', 2)

    g = evaluation_info["g"]
    g1 = evaluation_info["g1"]
    g2 = evaluation_info["g2"]

    Smokes = evaluation_info["Smokes"]
    Friends = evaluation_info["Friends"]
    Cancer = evaluation_info["Cancer"]

    smokes = evaluation_info["smokes"]
    friends = evaluation_info["friends"]
    cancer = evaluation_info["friends"]

    Not = evaluation_info["not"]
    Or = evaluation_info["or"]
    Implies = evaluation_info["implies"]
    Forall = evaluation_info["forall"]
    Exists = evaluation_info["exists"]

    df_smokes_cancer_facts = pd.DataFrame(np.array([[(x in smokes), (x in cancer) if x in g1 else math.nan] for x in g]),
        columns=["Smokes", "Cancer"],
        index=list('abcdefghijklmn'))
    df_friends_ah_facts = pd.DataFrame(np.array([[((x, y) in friends) if x < y else math.nan for x in g1] for y in g1]),
        index=list('abcdefgh'),
        columns=list('abcdefgh'))
    df_friends_in_facts = pd.DataFrame(
        np.array([[((x, y) in friends) if x < y else math.nan for x in g2] for y in g2]),
        index=list('ijklmn'),
        columns=list('ijklmn'))

    p = ltn.Variable.from_constants("p", list(g.values()))
    q = ltn.Variable.from_constants("q", list(g.values()))
    df_smokes_cancer = pd.DataFrame(tf.stack([Smokes(p).tensor, Cancer(p).tensor], axis=1).numpy(),
        columns=["Smokes", "Cancer"],
        index=list('abcdefghijklmn'))

    pred_friends = Friends([p, q]).tensor
    df_friends_ah = pd.DataFrame(pred_friends[:8, :8].numpy(), index=list('abcdefgh'), columns=list('abcdefgh'))
    df_friends_in = pd.DataFrame(pred_friends[8:, 8:].numpy(), index=list('ijklmn'), columns=list('ijklmn'))

    # Satisfiability of the axioms.

    # Store the printed output in a string
    output_string = ""
    output_string += "forall p: ~Friends(p,p) : %.2f\n" % Forall(p, Not(Friends([p, p]))).tensor.numpy()
    output_string += "forall p,q: Friends(p,q) -> Friends(q,p) : %.2f\n" % Forall((p, q), Implies(Friends([p, q]),Friends([q, p]))).tensor.numpy()
    output_string += "forall p: exists q: Friends(p,q) : %.2f\n" % Forall(p, Exists(q, Friends([p, q]))).tensor.numpy()
    output_string += "forall p,q: Friends(p,q) -> (Smokes(p)->Smokes(q)) : %.2f\n" % Forall((p, q), Implies(Friends([p, q]), Implies(Smokes(p),Smokes(q)))).tensor.numpy()
    output_string += "forall p: Smokes(p) -> Cancer(p) : %.2f\n" % Forall(p, Implies(Smokes(p), Cancer(p))).tensor.numpy()

    # We can query unknown formulas.
    output_string += "forall p: Cancer(p) -> Smokes(p): %.2f\n" % Forall(p, Implies(Cancer(p), Smokes(p)), p=5).tensor.numpy()
    output_string += "forall p,q: (Cancer(p) or Cancer(q)) -> Friends(p,q): %.2f\n" % Forall((p, q),Implies(Or(Cancer(p),Cancer(q)),Friends([p, q])),p=5).tensor.numpy()

    print(output_string)
    # Save the string to a file
    with open(path_to_results + "evaluation_results.txt", 'w') as out_file:
        out_file.write(output_string)

    plots_smokes_friend_cancer_facts(
        {"df_smokes_cancer_facts": df_smokes_cancer_facts , "df_friends_ah_facts":  df_friends_ah_facts, "df_friends_in_facts": df_friends_in_facts},
        path_to_results + "smokes_friend_cancer_facts_before_training.png"
    )

    # Querying all the truth-values using LTN after training:
    plots_smokes_friend_cancer_facts(
        {"df_smokes_cancer_facts": df_smokes_cancer, "df_friends_ah_facts": df_friends_ah, "df_friends_in_facts": df_friends_in},
        path_to_results + "smokes_friend_cancer_facts_after_training.png"
    )

    visualize_embeddings(g, g1, g2, Smokes, Friends, Cancer, path_to_results)