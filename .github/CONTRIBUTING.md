# Contributing to PyTorch3D
We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests
We actively welcome your pull requests.

However, if you're adding any significant features, please make sure to have a corresponding issue to outline your proposal and motivation and allow time for us to give feedback, *before* you send a PR.
We do not always accept new features, and we take the following factors into consideration:

- Whether the same feature can be achieved without modifying PyTorch3D directly. If any aspect of the API is not extensible, please highlight this in an issue so we can work on making this more extensible.
- Whether the feature is potentially useful to a large audience, or only to a small portion of users.
- Whether the proposed solution has a good design and interface.
- Whether the proposed solution adds extra mental/practical overhead to users who don't need such feature.
- Whether the proposed solution breaks existing APIs.

When sending a PR, please ensure you complete the following steps:

1. Fork the repo and create your branch from `master`. Follow the instructions
   in [INSTALL.md](../INSTALL.md) to build the repo.
2. If you've added code that should be tested, add tests.
3. If you've changed any APIs, please update the documentation.
4. Ensure the test suite passes:
    ```
    cd pytorch3d/tests
    python -m unittest -v
    ```
5. Make sure your code lints by running `dev/linter.sh` from  the project root.
6. If a PR contains multiple orthogonal changes, split it into multiple separate PRs.
7. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style  
We follow these [python](http://google.github.io/styleguide/pyguide.html) and [C++](https://google.github.io/styleguide/cppguide.html) style guides.

For the linter to work, you will need to install `black`, `flake`, `isort` and `clang-format`, and
they need to be fairly up to date.

## License
By contributing to PyTorch3D, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
