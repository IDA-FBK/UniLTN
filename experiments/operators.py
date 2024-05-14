import ltn


def build_operator(type_op, operator, parameters={}):
    """
    Build a logical operator based on the specified type and operator.

    Parameters:
    - type_op (str): Type of operator to build. Currently supported types: "ltn".
    - operator (str): The specific operator to build.
    - parameters (dict, optional): Additional parameters for building the operator.

    Returns:
    - object or None: The built operator object, or None if the specified type or operator is not supported.
    """

    built_operator = None

    if type_op == "ltn":
        if operator == "not":
            built_operator = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
        elif operator == "and":
            built_operator = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
        elif operator == "or":
            built_operator = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
        elif operator == "implies":
            built_operator = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
        elif operator == "equiv":
            And = None
            Implies = None
            if "prod" in parameters:
                And = ltn.fuzzy_ops.And_Prod()
            if "reichen" in parameters:
                Implies = ltn.fuzzy_ops.Implies_Reichenbach()
            built_operator = ltn.Wrapper_Connective(ltn.fuzzy_ops.Equiv(And, Implies))
        elif operator == "forall":
            if "p" not in parameters:
                built_operator = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(), semantics="forall")
            else:
                built_operator = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(parameters["p"]), semantics="forall")
        elif operator == "exists":
            if "p" not in parameters:
                built_operator = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(), semantics="exists")
            else:
                built_operator = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(parameters["p"]), semantics="exists")
        elif operator == "fagg":
            if "mean" in parameters:
                if "p" not in parameters:
                    built_operator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_Mean())
                else:
                    built_operator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_Mean(parameters["p"]))
            elif "pmean" in parameters:
                if "p" not in parameters:
                    built_operator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError())
                else:
                    built_operator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(parameters["p"]))
    elif type_op == "uniltn":
        if operator == "not":
            built_operator = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
        elif operator == "and":
            built_operator = ltn.Wrapper_Connective(ltn.fuzzy_ops.UniNormOperator(0.99,ltn.fuzzy_ops.And_Prod(),ltn.fuzzy_ops.Or_ProbSum()))
        elif operator == "or":
            built_operator = ltn.Wrapper_Connective(ltn.fuzzy_ops.UniNormOperator(0.1,ltn.fuzzy_ops.And_Prod(),ltn.fuzzy_ops.Or_ProbSum()))
        elif operator == "implies":
            built_operator = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_QL())
        elif operator == "equiv":
            And = None
            Implies = None
            if "prod" in parameters:
                And = ltn.fuzzy_ops.And_Prod()
            if "reichen" in parameters:
                Implies = ltn.fuzzy_ops.Implies_Reichenbach()
            built_operator = ltn.Wrapper_Connective(ltn.fuzzy_ops.Equiv(And, Implies))
        elif operator == "forall":
            if "p" not in parameters:
                built_operator = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(), semantics="forall")
            else:
                built_operator = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(parameters["p"]), semantics="forall")
        elif operator == "exists":
            if "p" not in parameters:
                built_operator = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.UninormQuantifier(g=0.8),semantics="exists")
            else:
                built_operator = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(parameters["p"]), semantics="exists")
        elif operator == "fagg":
            if "mean" in parameters:
                if "p" not in parameters:
                    built_operator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_Mean())
                else:
                    built_operator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_Mean(parameters["p"]))
            elif "pmean" in parameters:
                if "p" not in parameters:
                    built_operator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError())
                else:
                    built_operator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(parameters["p"]))

    return built_operator


def get_operators(operators):
    """
    Get a dictionary of logical operators based on the provided configuration.

    Parameters:
    - operators (dict): A dictionary where keys represent operator information (type-op) and values represent additional parameters.

    Returns:
    - dict: A dictionary containing logical operator objects, with operator names as keys.
    """
    tmp_operators = {}

    for operators_info, parameters in operators.items():
        type_op, op = operators_info.split("-")
        tmp_operators[op] = build_operator(type_op, op, parameters)

    return tmp_operators