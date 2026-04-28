from types import SimpleNamespace

from deerflow.community.aio_sandbox.local_backend import LocalContainerBackend


def test_start_container_honors_loopback_bind_host(monkeypatch):
    backend = LocalContainerBackend(
        image="example:image",
        base_port=39080,
        container_prefix="test-sandbox",
        config_mounts=[],
        environment={},
    )
    backend._runtime = "docker"

    captured = {}

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return SimpleNamespace(stdout="container-id\n")

    monkeypatch.setenv("DEER_FLOW_SANDBOX_BIND_HOST", "127.0.0.1")
    monkeypatch.setattr("subprocess.run", fake_run)

    container_id = backend._start_container("test-sandbox-1", 39080)

    assert container_id == "container-id"
    assert "-p" in captured["cmd"]
    port_index = captured["cmd"].index("-p") + 1
    assert captured["cmd"][port_index] == "127.0.0.1:39080:8080"
