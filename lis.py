import re
import sys
import operator
import pprint as pretty_print
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
        return self.value

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
    def __init__(self, series):
        self.denominator = series.count()
        self.numerator = series.apply(lambda x: x).count()
        self.proportion = self.numerator / self.denominator
        self.variance = (self.proportion * (1 - self.proportion)) / self.denominator
        self.mean = series.count() * series.mean()
        self.good = True

    def multiply_constant(self, const):
        self.mean = self.mean * const
        self.variance = self.mean * (const**2)
        return self

    def __repr__(self):
        return "Variable: {}".format(self.__dict__.__str__())


def add_series(name, series, env):
    env[name] = Variable(series)
    return env

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
    if isinstance(x, int):
        return y.multiply_constant(x)
    elif isinstance(y, int):
        return x.multiply_constant(y)


variable_environment = {
    '*': variable_multiplication
}

def variables_in_args(args):
    return any([isinstance(x, Variable) for x in args])

def eval(expr, environment, variable=False):
    print('IN EVAL: GOT EXPR {}'.format(expr))
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
            print('  Got a symbol {}'.format(environment[expr.value]))
            return environment[expr.value]
        else:
            print('we are getting a variable operator for {}'.format(expr))
            return variable_environment[expr.value]

    elif isinstance(expr, list):

        '''
            Most of the language’s built-ins are defined as Python code but we need to handle some language constructs
            like lambda or if directly in the interpreter, because they require a specific evaluation order.
            For example if we defined if as a function, this expression (if (= 3 2) (print '3 = 2') (print '3 = 3'))
            would print both 3 = 2 and 3 = 3, because the eval function evaluates its arguments in order.
        '''

        # print('IT IS A LIST!!!!')
        print('LIST LENGTH: {}'.format(len(expr)))
        if len(expr) == 0:
            print('Returning NONE')
            return None

        # At least 3 elements
        # Odd number of elements
        # All of the interstitial elements are infixable
        if all([
            len(expr) >= 3,
            len(expr) % 2 == 1,
            set([x.__str__() for x in expr[1::2]]).issubset(set(infix_operators))
        ]):
            print('We got a case of the infixes!')
            # First, do the mult operators, from left to right
            # Next, do the add_operators, from left to right
            # Finally, do the comparator operators, I guess, left to right
            operators = [x.__str__() for x in expr[1::2]]
            for oper_list in operator_hierarchy:
                # print('oper_list {}'.format(oper_list))
                for ind in range(len(operators)):
                    if operators[ind] in oper_list:
                        expr_strs = [x.__str__() for x in expr]
                        # print('Found {} first'.format(operators[ind]))
                        # Do this one
                        oper_index = find_first_index(operators[ind], expr_strs)
                        # print('.    oper_index {}'.format(oper_index))
                        # print('.    expr[oper_index] {}'.format(expr[oper_index]))
                        # print('.    actual_operator {}'.format(actual_operator))
                        first_arg = eval(expr[oper_index - 1], environment) if type(expr[oper_index - 1]) in [list, Symbol] else expr[oper_index - 1]
                        second_arg = eval(expr[oper_index + 1], environment) if type(expr[oper_index + 1]) in [list, Symbol] else expr[oper_index + 1]

                        actual_operator = eval(
                            expr[oper_index],
                            environment,
                            variables_in_args([first_arg, second_arg]))
                        print('    Running {} on {} and {}'.format(expr[oper_index], first_arg, second_arg))

                        new_field = apply(
                            actual_operator,
                            [first_arg, second_arg],
                            environment)
                        # print('Got back {}'.format(new_field))
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
                # Run inside of following list
                print('Defined! Now running the rest of list {}'.format(expr[3:][0]))
                return eval(expr[3:][0], environment)

            elif expr[0].value == 'begin':
                for ex in expr[1:]:
                    eval(ex, environment)

            else:
                print('we are in the list else {}'.format(expr))
                args = [eval(arg, environment) for arg in expr[1:]]
                fn = eval(
                    expr[0],
                    environment,
                    variables_in_args(args))
                print('now applying my list else ==> {} to {}'.format(fn, args))
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
