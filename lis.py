import re
import sys
import operator
import math
import scipy.stats as stats
import pprint as pretty_print
from pampy import match, _

pprint = lambda obj: pretty_print.PrettyPrinter(indent=4).pprint(obj)
infix_operators = ['*', '+', '/', '-', '==', '>', '<']
comp_operators = ['==', '>', '<']
mult_operators = ['*', '/']
add_operators = ['+', '-']

operator_hierarchy = [
    ['*', '/'],
    ['+', '-'],
    ['==', '>', '<']
]

def add_ns(a, b):
    if a is None or b is None:
        return None
    return a + b

def find_first_index(el, lst, ind=0):
    if len(lst) == 0:
        return False
    elif lst[0] == el:
        return ind
    return find_first_index(el, lst[1:], ind + 1)

def fail(s):
    print(s)
    sys.exit(-1)

class InterpreterObject(object):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return str(self.value)

class Symbol(InterpreterObject):
    pass

class String(InterpreterObject):
    pass

class Lambda(InterpreterObject):
    def __init__(self, arguments, code):
        self.arguments = arguments
        self.code = code

    def __repr__(self):
        return "(lambda (%s) (%s)" % (self.arguments, self.code)

class Variable(InterpreterObject):
    pass


class P_Value(Variable):
    def __init__(self, p_value):
        self.p_value = p_value

    def __repr__(self):
        return "[P-Value: {}]".format(self.p_value)

class Probability(Variable):
    def __init__(self, probability, p_value=False):
        self.probability = probability
        self.p_value = p_value

    def __repr__(self):
        if not self.p_value:
            return "[PROBABILITY: {}]".format(self.probability)
        elif self.probability <= 0.05:
            return "P-Value {} :: 🙅 - Reject the Null".format(self.probability)
        else:
            return "P-Value {} :: 👍 - Null Hypothesis Confirmed".format(self.probability)

    def multiply_constant(self, value):
        return self.probability * value

    def equals(self, y):
        bool_value = 1 if self.probability == y.probability else 0
        return bool_value

class Distribution(Variable):
    """ This will be for something like NPS """
    def __init__(self, mean, variance, n=None):
        self.mean = mean
        self.variance = variance
        self.n = n

    def __repr__(self):
        return "[DISTRIBUTION: mean={}, var={}]".format(self.mean, self.variance)

    def less_than(self, value):
        std_dev = math.sqrt(self.variance)
        z_value = (value - self.mean) / std_dev
        return Probability(stats.norm.cdf(z_value))

    def greater_than(self, value):
        std_dev = math.sqrt(self.variance)
        z_value = (self.mean - value) / std_dev
        return Probability(stats.norm.cdf(z_value))

    def multiply_constant(self, value):
        new_mean = self.mean * value
        new_var = self.variance * (value**2)
        return Distribution(new_mean, new_var, n=self.n)

    def subtract_proportion(self, prop):
        new_mean = self.mean - prop.mean
        new_var = self.variance + prop.variance + (2 * self.mean * prop.mean)
        new_n = add_ns(self.n, prop.n)
        return Distribution(new_mean, new_var, n=new_n)

    def add_proportion(self, prop):
        new_mean = self.mean + prop.mean
        new_var = self.variance + prop.variance - (2 * self.mean * prop.mean)
        new_n = add_ns(self.n, prop.n)
        return Distribution(new_mean, new_var, n=new_n)

    def equals(self, y):
        bool_value = 1 if self.mean == y.mean and self.variance == y.variance else 0
        return bool_value

    def test(self, y):
        z_score = (self.mean - y.mean) / math.sqrt((self.variance / self.n) + (y.variance / y.n))
        return Probability(stats.norm.cdf(z_score), p_value=True)


class Proportion(Distribution):
    def __init__(self, **kwargs):
        if 'series' in kwargs.keys():
            series = kwargs['series']
            self.denominator = series.count()
            self.n = series.count()
            self.numerator = series[series != 0].count()
            self.p = self.numerator / self.denominator
            self.mean = series.mean()
            self.variance = series.var()
        else:
            self.numerator = kwargs['numerator']
            self.denominator = kwargs['denominator']
            self.n = kwargs['denominator']
            self.mean = self.numerator
            self.p = self.numerator / self.denominator
            self.variance = self.p * (1 - self.p)


    def __repr__(self):
        return "[PROPORTION: {} / {}]".format(self.numerator, self.denominator)

    def greater_than(self, prop):
        sub_prop = self.subtract_proportion(prop)
        return sub_prop.greater_than(0)

    def less_than(self, prop):
        sub_prop = self.subtract_proportion(prop)
        return sub_prop.less_than(0)


class Value(Variable):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "VALUE: {}".format(self.value)

    def equals(self, y):
        bool_value = 1 if self.value == y.value else 0
        return bool_value

class Fail(Variable):
    def __repr__(self):
        return "FAILURE!!"

class VariableFactory:
    def __call__(self, **kwargs):
        return match(kwargs,
            {'series': _ }, lambda s: Proportion(**{'series': s}),
            {'numerator': _, 'denominator': _}, lambda n, d: Proportion(n, d),
            {'value': _}, lambda x: Value(x),
            {'mean': _, 'variance': _}, lambda m, v: Distribution(m, v),
            _, lambda x: Fail(None)
        )

variable_factory = VariableFactory()

def tokenize(s):
    ret = []
    in_string = False
    current_word = ''

    for i, char in enumerate(s):
        if char == "'":
            if in_string is False:
                in_string = True
                current_word += char
            else:
                in_string = False
                current_word += char
                ret.append(current_word)
                current_word = ''

        elif in_string is True:
            current_word += char

        elif char in ['\t', '\n', ' ']:
            continue

        elif char in ['(', ')']:
            ret.append(char)

        else:
            current_word += char
            if i < len(s) - 1 and s[i+1] in ['(', ')', ' ', '\n', '\t']:
                ret.append(current_word)
                current_word = ''

    return ret

'''
    The algorithm is simple: we scan the string char by char and look if we’re at a word, string or s-expr boundary.
    Depending on the case, we either add a new char to the current word, create a new word or a new sublist.
    The algorithm is complicated by the fact that we have several delimiters: spaces, simple quotes and braces,
    which makes it some sort of weird state machine.
'''



'''
    Next are a handful of utility functions that will help us convert tokens to their actual values.
'''



def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_string(s):
    if s[0] == "'" and s[-1] == "'":
        return True

    return False

'''
    The parse function is actually a wrapper for do_parse –
    I wanted to pass an iterator around because it
    felt a lot nicer than passing raw array indices around.
'''


def parse(tokens):
    itert = iter(tokens)
    token = next(itert)

    if token != '(':
        fail("Unexpected token {}".format(token))

    return do_parse(itert)


def do_parse(tokens):
    ret = []

    current_sexpr = None
    in_sexp = False

    for token in tokens:
        if token == '(':
            ret.append(do_parse(tokens))
        elif token == ')':
            return ret
        elif is_integer(token):
            ret.append(int(token))
        elif is_float(token):
            ret.append(float(token))
        elif is_string(token):
            ret.append(String(token[1:][0:-1]))
        else:
            ret.append(Symbol(token))


'''
    INTERPRETER:
    The interpreter is broken into two functions, eval and apply.
    Both take an s-expression (expr)
        and a dictionary of variables in scope (environment) as parameters.
    Eval‘s role is to take an expression and return its value.
    For example, if you pass a symbol to eval, it will look up its value in the symbol table and return it.
    apply is reserved for evaluating functions. It takes as parameters a function (written in Lisp or Python),
    a list of arguments and calls the function.
    How does it do that?
        It simply updates the environment
        to define the function’s parameters as local variables,
        and then calls eval!
'''
# First, let’s define how to evaluate numbers, strings and Symbols.

def variable_multiplication(x, y):
    return match([x, y],
        [Distribution, int], lambda x, y: x.multiply_constant(y),
        [Distribution, float], lambda x, y: x.multiply_constant(y),
        [int, Distribution], lambda x, y: y.multiply_constant(x),
        [float, Distribution], lambda x, y: y.multiply_constant(x),

        [Probability, int], lambda x, y: x.multiply_constant(y),
        [Probability, float], lambda x, y: x.multiply_constant(y),
        [int, Probability], lambda x, y: y.multiply_constant(x),
        [float, Probability], lambda x, y: y.multiply_constant(x),
        [_, _], lambda x, y: variable_factory(failure=1)
    )

def variable_minus(x, y):
    return match([x, y],
        [Distribution, Distribution], lambda x, y: x.subtract_proportion(y),
        [_, _], lambda x, y: variable_factory(failure=1)
    )

def variable_plus(x, y):
    return match([x, y],
        [Distribution, Distribution], lambda x, y: x.add_proportion(y),
        [_, _], lambda x, y: variable_factory(failure=1)
    )

def variable_greater_than(x, y):
    return match([x, y],
        [Proportion, Proportion], lambda x, y: x.greater_than(y),
        [_, _], lambda x, y: variable_factory(failure=1)
    )

def variable_less_than(x, y):
    return match([x, y],
        [Proportion, Proportion], lambda x, y: x.less_than(y),
        [_, _], lambda x, y: variable_factory(failure=1)
    )

def variable_equals(x, y):
    return match([x, y],
        [Distribution, Distribution], lambda x, y: x.equals(y),
        [Probability, Probability], lambda x, y: x.equals(y),
        [Value, Value], lambda x, y: x.equals(y),
        [_, _], lambda x, y: variable_factory(failure=1)
    )

def variable_test(x, y):
    return match([x, y],
        [Distribution, Distribution], lambda x, y: x.test(y),
        [_, _], lambda x, y: variable_factory(failure=1)
    )

variable_environment = {
    '-': variable_minus,
    '+': variable_plus,
    '*': variable_multiplication,
    '>': variable_greater_than,
    '<': variable_less_than,
    '==': variable_equals,
    'test': variable_test
}

def variables_in_args(args):
    return any([isinstance(x, Variable) for x in args])

def eval(expr, environment, variable=False):
    print('λ {}'.format(expr))
    # print('OF TYPE: {}'.format(type(expr)))
    if isinstance(expr, int):
        return expr
    elif isinstance(expr, str):
        return expr
    elif isinstance(expr, float):
        return expr
    elif isinstance(expr, String):
        return expr.value
    elif isinstance(expr, Variable):
        return expr
    elif isinstance(expr, Symbol):
        if not variable:
            if expr.value not in environment:
                print('OOPS')
                print(expr.value)
                print(environment)
                print(Symbol)
                fail("Couldn't find symbol {}".format(expr.value))
            print('  {}'.format(environment[expr.value]))
            return environment[expr.value]
        else:
            print('📊 Stat Variable Operator {}'.format(expr))
            return variable_environment[expr.value]

    elif isinstance(expr, list):

        '''
            Most of the language’s built-ins are defined as Python code but we need to handle some language constructs
            like lambda or if directly in the interpreter, because they require a specific evaluation order.
            For example if we defined if as a function, this expression (if (= 3 2) (print '3 = 2') (print '3 = 3'))
            would print both 3 = 2 and 3 = 3, because the eval function evaluates its arguments in order.
        '''
        print('LIST LENGTH: {}'.format(len(expr)))
        if len(expr) == 0:
            print('😢')
            return None

        # At least 3 elements
        # Odd number of elements
        # All of the interstitial elements are infixable
        if all([
            len(expr) >= 3,
            len(expr) % 2 == 1,
            set([x.__str__() for x in expr[1::2]]).issubset(set(infix_operators))
        ]):
            # First, do the mult operators, from left to right
            # Next, do the add_operators, from left to right
            # Finally, do the comparator operators, I guess, left to right
            operators = [x.__str__() for x in expr[1::2]]
            for oper_list in operator_hierarchy:
                for ind in range(len(operators)):
                    if operators[ind] in oper_list:
                        expr_strs = [x.__str__() for x in expr]

                        # Do this one
                        oper_index = find_first_index(operators[ind], expr_strs)
                        first_arg = eval(expr[oper_index - 1], environment) if type(expr[oper_index - 1]) in [list, Symbol] else expr[oper_index - 1]
                        second_arg = eval(expr[oper_index + 1], environment) if type(expr[oper_index + 1]) in [list, Symbol] else expr[oper_index + 1]

                        actual_operator = eval(
                            expr[oper_index],
                            environment,
                            variables_in_args([first_arg, second_arg]))
                        print('   👷Running ({}) on {} and {}'.format(expr[oper_index], first_arg, second_arg))

                        new_field = apply(
                            actual_operator,
                            [first_arg, second_arg],
                            environment)

                        expr[(oper_index - 1): (oper_index + 2 )] = [new_field]
                        print('         Now Expression is', expr)

            return eval(expr[0], environment)

        ''' Lambda is simply an object that holds arguments (a list of arguments with their names)
            and code (a list of Lisp instructions to execute). '''
        if isinstance(expr[0], Symbol):
            if expr[0].value == 'lambda':
                arg_names = expr[1]
                code = expr[2]

                return Lambda(arg_names, code)

            elif expr[0].value == 'if':
                condition = expr[1]
                then = expr[2]
                _else = None
                if len(expr) == 4:
                    _else = expr[3]

                if eval(condition, environment) != False:
                    return eval(then, environment)
                elif _else is not None:
                    return eval(_else, environment)

            elif expr[0].value == 'define':
                name = expr[1].value
                value = eval(expr[2], environment)
                environment[name] = value

            elif expr[0].value == 'begin':
                for ex in expr[1:-1]:
                    eval(ex, environment)

                return eval(expr[-1], environment)

            else:
                args = [eval(arg, environment) for arg in expr[1:]]
                fn = eval(
                    expr[0],
                    environment,
                    variables_in_args(args))
                print('🏃 Applying List Else ==> {} to {}'.format(fn, args))
                return apply(fn, args, environment)

#Apply is pretty simple too. It checks if a function is an interpreter built-in or not.


def apply(fn, args, environment):
    # If it is, it simply passes the arguments to the built-in.

    if callable(fn):
        return fn(*args)

    # Otherwise we have to actually evaluate the function.

    if isinstance(fn, Lambda):
        new_env = dict(environment)
        if len(args) != len(fn.arguments):
            fail("Mismatched number of arguments to lambda")

        # To do this we bind the values of the arguments to the environment.

        for i in range(len(fn.arguments)):
            new_env[fn.arguments[i].value] = args[i]

        print('IN LAMBDA APPLICATION, CODE :{}'.format(fn.code))
        print('IN LAMBDA APPLICATION, NEWENV :{}'.format(new_env))
        # And we call eval on the function body. That’s it!
        return eval(fn.code, new_env)

# Finally we define a handful of system built-ins.

base_environment = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
    '>': operator.gt,
    '>=': operator.ge,
    '<': operator.lt,
    '<=': operator.le,
    '=': operator.eq,
    '!=': operator.ne,
    'nil': None,
    'print': lambda x: sys.stdout.write(str(x) + '\n'),
}


def run(str, env=base_environment):
    return eval(parse(tokenize(str)), env)

def main():
    if len(sys.argv) != 2:
        print("usage: python {} <file>".format(sys.argv[0]))
        sys.exit(-1)

    with open(sys.argv[1]) as fd:
        contents = fd.read()
        parsed = parse(tokenize(contents))
        eval(parsed, base_environment)


if __name__ == '__main__':
    main()
