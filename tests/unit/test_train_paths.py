import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("qtpy")
pytest.importorskip("magicgui")

from das import gui_app as gui


@pytest.mark.parametrize("output_dir", ["", "   ", "/custom/models", "./"])
@pytest.mark.parametrize("selection", ["setter", "browse", "typed"])
def test_train_data_directory_defaults_only_empty_output(qtbot, tmp_path, monkeypatch, output_dir, selection):
    paths = gui.TrainPathsWidget()
    qtbot.addWidget(paths)
    paths.output_dir = output_dir

    if selection == "setter":
        paths.data_dir = str(tmp_path)
    elif selection == "browse":
        monkeypatch.setattr(gui.QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
        paths.data_dir_field.browse()
    else:
        qtbot.keyClicks(paths.data_dir_field.line_edit, str(tmp_path))
        assert paths.output_dir == output_dir.strip()
        paths.data_dir_field.line_edit.editingFinished.emit()

    expected = output_dir.strip() or str(tmp_path / "models")
    assert paths.output_dir == expected
    paths.data_dir = str(tmp_path / "other")
    assert paths.output_dir == expected


def test_empty_training_data_directory_leaves_output_empty(qtbot):
    paths = gui.TrainPathsWidget()
    qtbot.addWidget(paths)
    paths.data_dir = ""
    paths.data_dir_field.line_edit.editingFinished.emit()
    assert paths.output_dir == ""
