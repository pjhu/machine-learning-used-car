# read large file

## [memory_profiler](https://pypi.python.org/pypi/memory_profiler)

script:
```
from memory_profiler import profile

@profile
def my_func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    return a
```

run:
```
python -m memory_profiler example.py
```

output:
```
Line #    Mem usage  Increment   Line Contents
==============================================
     3                           @profile
     4      5.97 MB    0.00 MB   def my_func():
     5     13.61 MB    7.64 MB       a = [1] * (10 ** 6)
     6    166.20 MB  152.59 MB       b = [2] * (2 * 10 ** 7)
     7     13.61 MB -152.59 MB       del b
     8     13.61 MB    0.00 MB       return a
```

## [line_profiler](https://github.com/rkern/line_profiler)

script:
```
from line_profiler import LineProfiler
import random

def do_stuff(numbers):
    s = sum(numbers)
    l = [numbers[i]/43 for i in range(len(numbers))]
    m = ['hello'+str(numbers[i]) for i in range(len(numbers))]

numbers = [random.randint(1,100) for i in range(1000)]
lp = LineProfiler()
lp_wrapper = lp(do_stuff)
lp_wrapper(numbers)
lp.print_stats()
```

output:
```
Timer unit: 1e-06 s

Total time: 0.000649 s
File: <ipython-input-2-2e060b054fea>
Function: do_stuff at line 4

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     4                                           def do_stuff(numbers):
     5         1           10     10.0      1.5      s = sum(numbers)
     6         1          186    186.0     28.7      l = [numbers[i]/43 for i in range(len(numbers))]
     7         1          453    453.0     69.8      m = ['hello'+str(numbers[i]) for i in range(len(numbers))]
```

information for each line:

```
Line #: The line number in the file.
Hits: The number of times that line was executed.
Time: The total amount of time spent executing the line in the timer's units. In the header information before the tables, you will see a line "Timer unit:" giving the conversion factor to seconds. It may be different on different systems.
Per Hit: The average amount of time spent executing the line once in the timer's units.
% Time: The percentage of time spent on that line relative to the total amount of recorded time spent in the function.
Line Contents: The actual source code. Note that this is always read from disk when the formatted results are viewed, not when the code was executed. If you have edited the file in the meantime, the lines will not match up, and the formatter may not even be able to locate the function for display.
```