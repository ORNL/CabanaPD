# Contributing

If you'd like to contribute to CabanaPD, please open a [pull
request](https://help.github.com/articles/using-pull-requests/) with
`main` as the destination branch on the
[repository](https://github.com/ORNL/CabanaPD) and allow edits from
maintainers in the pull request.

Your pull request must pass tests, which includes using the coding
style from `.clang-format` (enforced with clang-format-14), and be
reviewed by at least one CabanaPD developer. Formatting can be applied
with `make format` within the build folder.

`pre-commit` is a useful tool for ensuring feature branches are ready for
review by running automatic checks locally before a commit is made.
[Installation details](https://pre-commit.com/#install) (once per system) and
[activation details](https://pre-commit.com/#usage) (once per repo) are
available.
