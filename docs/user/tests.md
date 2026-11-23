---
icon: lucide/list-checks
---
# Building the tests

Unit tests can be built by updating the CabanaPD CMake configuration in the
script above with:

```
-D CabanaPD_ENABLE_TESTING=ON
```

GTest is required for CabanaPD unit tests, with build instructions
[here](https://github.com/google/googletest). If tests are enabled, you can run
the CabanaPD unit test suite with:

```
cd CabanaPD/build
ctest
```
