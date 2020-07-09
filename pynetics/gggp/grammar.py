# ======================================================================
# pynetics: a simple yet powerful evolutionary computation library
# Copyright (C) 2020 Alberto Díaz-Álvarez
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# “Software”), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
# THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ======================================================================
"""TODO TBD...

    Our grammar will be a weighted one (has the same expressiveness of a
    probabilistic one). Its definition (in Extended BNF) is as follows:

        grammar = rule { rule } ;

        rule = non_terminal , "->" , expression , ";" ;

        non_terminal = letter , { alphanum } ;

        alphanum = letter | digit | "_" ;

        expression = non_terminal
                   | terminal
                   | expression , "?"
                   | expression , "*"
                   | expression , "+"
                   | expression , "~" , integer , [".." , integer]
                   | expression , "[" , decimal , "]"
                   | "(" , expression , ")"
                   | expression , "|" , expression
                   | expression , "," , expression ;

        terminal = "'" , character , { character } , "'"
                 | '"' , character , { character } , '"' ;

        character = symbol | alphanum ;

        letter = "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I"
               | "J" | "K" | "L" | "M" | "N" | "O" | "P" | "Q" | "R"
               | "S" | "T" | "U" | "V" | "W" | "X" | "Y" | "Z" | "a"
               | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j"
               | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s"
               | "t" | "u" | "v" | "w" | "x" | "y" | "z" ;

        integer = digit , {digit};

        decimal = integer , ["." , integer];

        digit = "0" | "1" | "2" | "3" | "4"  | "5" | "6" | "7" | "8"
              | "9" ;

        symbol = "[" | "]" | "{" | "}" | "(" | ")" | "<" | ">" | "'"
               | '"' | "=" | "|" | "." | "," | ";" ;

    NOTE: White spaces are ignored
"""
import abc
import re


class Node(metaclass=abc.ABCMeta):
    pass


class NonTerminal(Node, metaclass=abc.ABCMeta):
    pass


class Rule(NonTerminal, metaclass=abc.ABCMeta):
    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent
        self.consequent = consequent


class Group(NonTerminal, metaclass=abc.ABCMeta):
    def __init__(self, node):
        self.node = node


class And(NonTerminal, metaclass=abc.ABCMeta):
    def __init__(self, *nodes):
        self.nodes = list(nodes)


class Or(NonTerminal, metaclass=abc.ABCMeta):
    def __init__(self, *nodes):
        self.nodes = list(nodes)


class ZeroToOne(NonTerminal, metaclass=abc.ABCMeta):
    def __init__(self, node):
        self.node = node


class ZeroToMany(NonTerminal, metaclass=abc.ABCMeta):
    def __init__(self, node):
        self.node = node


class OneToMany(NonTerminal, metaclass=abc.ABCMeta):
    def __init__(self, node):
        self.node = node


class Terminal(Node, metaclass=abc.ABCMeta):
    pass


class Literal(Terminal):
    def __init__(self, value):
        self.value = value


class GrammarParser:
    """A parser to process our grammar."""
    PATTERNS = r'''
        (?P<blank>\s+)
        |(?P<implies>->)
        |(?P<letter>[a-zA-Z])
        |(?P<underscore>_)
        |(?P<digit>\d)
        |(?P<symbol>[\[,\],\{,\},\(,\),\<,\>,\',\",\=,\|,\.,\,,\;])
    '''

    def __init__(self, text):
        self.text = text
        self.tokens = None
        self.grammar = None

        self.__tokenize()
        self.__parse_grammar()

    def __tokenize(self):
        compiler = re.compile(self.PATTERNS, re.VERBOSE | re.DOTALL)
        self.tokens = []
        pos = 0
        while True:
            m = compiler.match(self.text, pos)
            if m:
                pos = m.end()
                token_name = m.lastgroup
                token_value = m.group(token_name)
                self.tokens.append((token_name, token_value))
            else:
                break
        if pos != len(self.text):
            raise SyntaxError(f'Syntax error at position {pos}')
        else:
            self.tokens = [(c, t) for c, t in self.tokens if c != 'blank']

    def __parse_grammar(self):
        """TODO TBD

        grammar = rule , {rule} ;
        """
        self.grammar = [self.__parse_rule()]
        while self.tokens:
            self.grammar.append(self.__parse_rule())

    def __parse_rule(self):
        """TODO TBD...

        rule = non_terminal , "->" , expression , ";" ;
        """
        non_terminal = self.parse_non_terminal()
        self.pop('implies')
        expression = self.__parse_expression()
        self.pop('symbol', ';')

        return Rule(non_terminal, expression)

    def __parse_non_terminal(self):
        """TODO TBD...

        non_terminal = letter , { alphanum } ;
        """
        pass

    '''

        alphanum = letter | digit | "_" ;

        expression = non_terminal
                   | terminal
                   | expression , "?"
                   | expression , "*"
                   | expression , "+"
                   | expression , "~" , integer , [".." , integer]
                   | expression , "[" , decimal , "]"
                   | "(" , expression , ")"
                   | expression , "|" , expression
                   | expression , "," , expression ;

        terminal = "'" , character , { character } , "'"
                 | '"' , character , { character } , '"' ;

        character = symbol | alphanum ;

        integer = digit , {digit};

        decimal = integer , ["." , integer];

    def __parse_expression(self):
        """TODO TBD

        expression = "'" , literal , "'"
                   | '"' , literal , '"'
                   | expression , ',' , expression
                   | expression , '|' , expression
                   | '(' , expression , ')'
                   | expression , '?'
                   | expression , '*'
                   | expression , '+'
        """
        if self.head('single_quote'):
            self.pop('single_quote')
            result = Literal(self.pop('literal'))
            self.pop('single_quote')
        elif self.head('double_quote'):
            self.pop('double_quote')
            result = Literal(self.pop('literal'))
            self.pop('double_quote')
        elif self.head('open_group'):
            self.pop('open_group')
            result = Group(self.__parse_expression())
            self.pop('close_group')
        else:
            raise SyntaxError(f'Unexpected token: {self.tokens[0]}')

        if self.head('and'):
            self.pop('and')
            next_expression = self.__parse_expression()
            if isinstance(next_expression, And):
                nodes = [result] + next_expression.nodes
            else:
                nodes = [result, next_expression]
            result = And(*nodes)
        elif self.head('or'):
            self.pop('or')
            next_expression = self.__parse_expression()
            if isinstance(next_expression, Or):
                nodes = [result] + next_expression.nodes
            else:
                nodes = [result, next_expression]
            result = Or(*nodes)
        elif self.head('zero_to_one'):
            self.pop('zero_to_one')
            result = ZeroToOne(result)
        elif self.head('zero_to_many'):
            result = ZeroToMany(result)
        elif self.head('one_to_many'):
            result = OneToMany(result)

        return result

    def head(self, cls, expected=None):
        """Checks if the next token of tokens is of a specified class.

        Given a token type, the method will return whether or not the
        next element of the tokens stack is of the specified class. If
        there are no elements, then the method will return false.

        When head is called, the stack will not be modified.

        :param cls: the class to check.
        :param expected: The value to compare against. If None, no
            comparison is performed.
        :returns: True if the next item belongs to the expected class
            and has the expected value (if any), and False otherwise.
        """
        if self.tokens:
            token_cls, token_value = self.tokens[0]
            same_class = token_cls == cls
            same_value = expected is None or expected == token_value
            return same_class and same_value
        else:
            return False

    def pop(self, cls, expected=None):
        """Pops the next element iff belongs to the specified class.

        Given a class, the method will return the next element of the
        stack iff that element belongs to the specified class. In that's
        not the case, the method will fail and an error will be raised.

        When pop is called, the stack will be modified, removing the top
        element of the stack.

        :param cls: the class to check.
        :param expected: the value to compare against. If None, no
            comparison is performed.
        :returns: the element on the top of the stack.
        :raises SyntaxError: If the next element of the stack hasn't the
            specified token type, if it's different to the expected
            value (if provided), or if there are no more rules.
        """
        if len(self.tokens) > 0:
            token_cls, token_value = self.tokens.pop(0)
            if token_cls != cls:
                raise SyntaxError(cls, token_cls, token_value)
            elif expected is not None and expected != token_value:
                raise SyntaxError(f'Expected {expected} but got
                {token_value}')
            else:
                return token_value
        else:
            raise SyntaxError('Unexpected end of grammar')


equation =
    equation -> variable = expression;
    variable -> letter, (letter | digit)* ;
    expression -> variable
                | number
                | not , expression
                | expression operator expression;
    operator -> '+'
              | '-
    letter = "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I" | "J"
           | "K" | "L" | "M" | "N" | "O" | "P" | "Q" | "R" | "S" | "T"
           | "U" | "V" | "W" | "X" | "Y" | "Z" | "a" | "b" | "c" | "d"
           | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m" | "n"
           | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x"
           | "y" | "z" ;
        number -> digit+
    digit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;

g = GrammarParser(dummy_grammar)
print(g.tokens)
'''
