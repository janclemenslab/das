import pytest


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
    for widget in app.topLevelWidgets():
        widget.close()
    app.processEvents()


def test_gui_startup(app):
    from xarray_behave.gui.app import MainWindow

    window = MainWindow()
    window.show()
    app.processEvents()
    assert window.isVisible()


@pytest.mark.parametrize("tab,field", [("train", "data_dir"), ("predict", "model_path")])
def test_gui_das_dialogs(app, monkeypatch, tmp_path, tab, field):
    from PySide6 import QtWidgets
    from xarray_behave.gui.app import MainWindow
    from xarray_behave.gui.formbuilder import YamlDialog

    model = str(tmp_path / "test_model.h5")
    (tmp_path / "test_params.yaml").write_text("{}")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *args, **kwargs: (model, ""))
    dialogs = []

    def reject(dialog):
        dialogs.append(dialog)
        return QtWidgets.QDialog.Rejected

    monkeypatch.setattr(YamlDialog, "exec_", reject)
    window = MainWindow()
    getattr(window, f"das_{tab}")()
    assert len(dialogs) == 1
    assert field in dialogs[0].form.fields
