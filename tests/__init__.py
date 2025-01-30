# Here so `from .conftest import FIXTURES` resolves. Without it
# pytest imports each test file as a top-level module and every
# one of those imports fails at collection time.
