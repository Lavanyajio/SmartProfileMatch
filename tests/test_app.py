import gradio as gr


def test_ui_builds(monkeypatch, tmp_path):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    import app
    assert isinstance(app.build_ui(), gr.Blocks)


def test_respond_handles_errors_gracefully(monkeypatch):
    import app

    def boom(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(app.pipeline, "answer", boom)
    history, cleared = app.respond("hello", [], 5)
    assert cleared == ""
    assert history[0] == {"role": "user", "content": "hello"}
    assert "provider down" in history[-1]["content"]
