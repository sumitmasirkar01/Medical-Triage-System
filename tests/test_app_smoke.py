"""Render the supported Streamlit empty state without starting a web server."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from scripts.create_demo_dicom import create_demo_ct


def button_with_label(rendered: AppTest, label: str):
    return next(button for button in rendered.button if button.label == label)


def metric_value(rendered: AppTest, label: str) -> str:
    return next(metric.value for metric in rendered.metric if metric.label == label)


def test_streamlit_app_executes_supported_workflow(tmp_path):
    app = Path(__file__).resolve().parents[1] / "app.py"
    rendered = AppTest.from_file(str(app)).run(timeout=30)

    assert not rendered.exception
    assert [tab.label for tab in rendered.tabs] == [
        "Add DICOM",
        "Active worklist",
        "Model provenance",
    ]
    assert len(rendered.file_uploader) == 1
    assert rendered.button[0].label == "Run inference and enqueue"
    assert any("No active cases" in item.value for item in rendered.info)

    fixture = tmp_path / "DEMO_CT.dcm"
    create_demo_ct(fixture)
    rendered.file_uploader[0].upload(
        fixture.name, fixture.read_bytes(), "application/dicom"
    )
    button_with_label(rendered, "Run inference and enqueue").click()
    rendered.run(timeout=30)

    assert not rendered.exception
    assert metric_value(rendered, "Active cases") == "1"
    assert metric_value(rendered, "Effective zone") == "RED"
    assert metric_value(rendered, "Threshold result") == "Positive"

    rendered.text_input("override-actor-DEMO_CT").set_value("qa-radiologist")
    rendered.selectbox("zone-DEMO_CT").set_value("GREEN")
    rendered.text_area("reason-DEMO_CT").set_value("UI workflow verification")
    button_with_label(rendered, "Apply override").click()
    rendered.run(timeout=30)

    assert not rendered.exception
    assert metric_value(rendered, "Effective zone") == "GREEN"

    rendered.text_input("clear-actor-DEMO_CT").set_value("qa-radiologist")
    rendered.text_input("clear-reason-DEMO_CT").set_value("Restore automatic routing")
    button_with_label(rendered, "Clear override").click()
    rendered.run(timeout=30)

    assert not rendered.exception
    assert metric_value(rendered, "Effective zone") == "RED"

    rendered.text_input("complete-actor-DEMO_CT").set_value("qa-radiologist")
    button_with_label(rendered, "Mark completed").click()
    rendered.run(timeout=30)

    assert not rendered.exception
    assert metric_value(rendered, "Active cases") == "0"
    assert any("No active cases" in item.value for item in rendered.info)
