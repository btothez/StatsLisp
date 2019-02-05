import re
import sys
import operator
import pprint as pretty_print
pprint = lambda obj: pretty_print.PrettyPrinter(indent=4).pprint(obj)

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
    The interpreter is broken into two functions, eval and apply. Both take an s-expression (expr)
    and a dictionary of variables in scope (environment) as parameters.
    eval‘s role is to take an expression and return its value.
    For example, if you pass a symbol to eval, it will look up its value in the symbol table and return it.
    apply is reserved for evaluating functions. It takes as parameters a function (written in Lisp or Python),
    a list of arguments and calls the function.
    How does it do that? It simply updates the environment to define the function’s parameters as local variables, and then calls eval!
'''

# First, let’s define how to evaluate numbers, strings and Symbols.


def eval(expr, environment):
    print('IN EVAL: GOT EXPR {}'.format(expr))
    if isinstance(expr, int):
        return expr
    elif isinstance(expr, str):
        return expr
    elif isinstance(expr, float):
        return expr
    elif isinstance(expr, String):
        return expr.value
    elif isinstance(expr, Symbol):
        if expr.value not in environment:
            print('OOPS')
            print(expr.value)
            print(environment)
            print(Symbol)
            fail("Couldn't find symbol {}".format(expr.value))
        return environment[expr.value]
    elif isinstance(expr, list):

        '''
            Most of the language’s built-ins are defined as Python code but we need to handle some language constructs
            like lambda or if directly in the interpreter, because they require a specific evaluation order.
            For example if we defined if as a function, this expression (if (= 3 2) (print '3 = 2') (print '3 = 3'))
            would print both 3 = 2 and 3 = 3, because the eval function evaluates its arguments in order.
        '''

        if isinstance(expr[0], Symbol):
            if expr[0].value == 'lambda':
                arg_names = expr[1]
                code = expr[2]

                '''
                    Lambda is simply an object that holds arguments (a list of arguments with their names)
                    and code (a list of Lisp instructions to execute).
                '''

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
                for ex in expr[1:]:
                    eval(ex, environment)

            else:
                fn = eval(expr[0], environment)
                args = [eval(arg, environment) for arg in expr[1:]]
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
