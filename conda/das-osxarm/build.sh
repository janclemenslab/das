# We need to turn pip index back on because Anaconda turns
# it off for some reason.
export PIP_NO_INDEX=False
export PIP_NO_DEPENDENCIES=False
export PIP_IGNORE_INSTALLED=False

# $PYTHON -m pip install tensorflow==2.15.1 tensorflow-metal==1.1.0 numpy==1.23.5
# $PYTHON -m pip install tensorflow==2.15.1 tensorflow-metal==1.1.0
$PYTHON -m pip install das --no-dependencies

