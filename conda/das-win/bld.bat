set PIP_NO_INDEX=False
set PIP_NO_DEPENDENCIES=False
set PIP_IGNORE_INSTALLED=False
@REM pip install "tensorflow==2.10.*"
pip install das==0.32.3 --no-dependencies
if errorlevel 1 exit 1
