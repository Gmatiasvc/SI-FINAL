"""Pruebas automatizadas de la interfaz Streamlit multipágina usando `AppTest`."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def test_streamlit_navigation_all_pages_without_exceptions() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception
    assert len(at.title) == 1
    assert at.title[0].value == "Inspección integral"

    for page_path, expected_title in (
        ("app_pages/asistente.py", "Asistente RAG"),
        ("app_pages/metricas.py", "Métricas y evaluación"),
        ("app_pages/trazas_logs.py", "Trazas y órdenes"),
    ):
        at.switch_page(page_path).run()
        assert not at.exception
        assert at.title[0].value == expected_title


def test_streamlit_inspection_form_submission_and_results() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception

    # Ejecutar diagnóstico con valores por defecto (Escenario normal)
    at.form_submit_button(key="btn_run_inspection").click().run()
    assert not at.exception
    assert len(at.metric) >= 4
    assert at.session_state["last_agent_state"] is not None
    assert at.session_state["last_agent_state"]["expected_energy_kwh"] > 0


def test_streamlit_rag_assistant_chat_flow() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.switch_page("app_pages/asistente.py").run()
    assert not at.exception

    at.chat_input(key="rag_chat_input").set_value(
        "¿Qué controles manuales se recomiendan ante baja generación y suciedad visible?"
    ).run()
    assert not at.exception
    msgs = at.session_state["chat_messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert len(msgs[1]["citations"]) >= 1
