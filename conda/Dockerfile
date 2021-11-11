FROM mambaorg/micromamba
COPY --chown=micromamba:micromamba das das
COPY --chown=micromamba:micromamba das-nogui das-nogui

RUN micromamba install -y -n base -c conda-forge \
       python=3.8 \
       boa anaconda-client conda-build conda-verify
RUN micromamba clean --all --yes
RUN conda config --set anaconda_upload yes

RUN conda mambabuild das-nogui -c conda-forge -c ncb -c anaconda --python 3.9 --user ncb; exit 0
RUN conda mambabuild das-nogui -c conda-forge -c ncb -c anaconda --python 3.8 --user ncb; exit 0
RUN conda mambabuild das-nogui -c conda-forge -c ncb -c anaconda --python 3.7 --user ncb; exit 0
RUN conda mambabuild das -c conda-forge -c ncb -c anaconda --python 3.9 --user ncb; exit 0
RUN conda mambabuild das -c conda-forge -c ncb -c anaconda --python 3.8 --user ncb; exit 0
RUN conda mambabuild das -c conda-forge -c ncb -c anaconda --python 3.7 --user ncb; exit 0
