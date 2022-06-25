import functools

def abc(a, b, c):
    print(f"{a}, {b}, {c}")


aaa = functools.partial(abc, b=1, c=2)

aaa(3)

def kww(a, b=1, **kwargs):
    print(f"{a}, {b}")
    for k, v in kwargs.items():
        print(f"{k}:{v}")

kww(1, **{'b':10, 'c':20})

class foo:
    def __init__(self):
        self.a=1
    def __call__(self, b):
        def fooFunc():
            def printa(b):
                print('a,b')
        return fooFunc.printa(b)

ff = foo()
ff(2)
