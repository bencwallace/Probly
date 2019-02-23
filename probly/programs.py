# Exec programs for automating repetitive operator definitions
_num_ops_lift = ['add', 'sub', 'mul', 'matmul',
                 'truediv', 'floordiv', 'mod', 'divmod', 'pow']
_num_ops_right = ['add', 'sub', 'mul', 'matmul', 'truediv', 'floordiv', 'mod',
                  'divmod', 'pow']
_num_ops_unary = ['neg', 'pos', 'abs', 'complex', 'int', 'float', 'round',
                  'trunc', 'floor', 'ceil']

_programs_lift = [
    (
        '@Lift\n'
        'def __{:s}__(self, x):\n'
        '   return op.{:s}(self, x)'
    ).format(fcn, fcn) for fcn in _num_ops_lift]

_programs_right = [
    (
        'def __r{:s}__(self, x):\n'
        '   X = rvar._cast(x)\n'
        '   return X.__{:s}__(self)'
    ).format(fcn, fcn) for fcn in _num_ops_right]

_programs_unary = [
    (
        '@Lift\n'
        'def __{:s}__(self):\n'
        '   return op.{:s}(self)'
    ).format(fcn, fcn) for fcn in _num_ops_unary]
_programs = _programs_lift + _programs_right + _programs_unary
