# We need to turn pip index back on because Anaconda turns
# it off for some reason.
export PIP_NO_INDEX=False
export PIP_NO_DEPENDENCIES=False
export PIP_IGNORE_INSTALLED=False

$PYTHON -m pip install keras-tuner -vv --no-dependencies
$PYTHON -m pip install kt-legacy -vv --no-dependencies
$PYTHON -m pip install tensorflow-macos==2.8 tensorflow-metal
$PYTHON -m pip install das -vv --no-dependencies
