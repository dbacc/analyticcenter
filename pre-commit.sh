#!/bin/sh
# to install it, create a symbolic link in the projects .git/hooks folder
#
#       i.e. - from the .git/hooks directory, run
#               $ ln -s ../../git-hooks/pre-commit.sh pre-commit
#
# to skip the tests, run with the --no-verify argument
#       i.e. - $ 'git commit --no-verify'

# stash any unstaged changes
git stash -q --keep-index

# run the tests
python setup.py test

# store the last exit code in a variable
RESULT=$?

# unstash the unstashed changes
git stash pop -q

# return the 'test' exit code
exit $RESULT
