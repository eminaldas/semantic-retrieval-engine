import io
import json

from backend.mcp import server as srv


class FakeService:
    def __init__(self):
        self.calls = []

    def search(self, query, project=None, doc_type=None, top_k=6):
        self.calls.append(("search", query, project, top_k))
        return f"sonuc: {query}"

    def add_note(self, title, body, project=None):
        raise ValueError("title ve body bos olamaz")

    def list_projects(self):
        raise RuntimeError("qdrant kapali")


def _call(server, method, params=None, msg_id=1):
    return server.handle({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}})


class TestHandshake:
    def test_initialize_echoes_supported_version(self):
        resp = _call(srv.MCPServer(FakeService()), "initialize", {"protocolVersion": "2025-03-26"})
        assert resp["result"]["protocolVersion"] == "2025-03-26"
        assert resp["result"]["capabilities"] == {"tools": {"listChanged": False}}

    def test_initialize_falls_back_to_latest_for_unknown_version(self):
        resp = _call(srv.MCPServer(FakeService()), "initialize", {"protocolVersion": "1999-01-01"})
        assert resp["result"]["protocolVersion"] == srv.SUPPORTED_PROTOCOL_VERSIONS[0]

    def test_notification_gets_no_response(self):
        assert srv.MCPServer(FakeService()).handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    def test_unknown_method(self):
        assert _call(srv.MCPServer(FakeService()), "resources/list")["error"]["code"] == -32601


class TestTools:
    def test_tools_list_names(self):
        names = [t["name"] for t in _call(srv.MCPServer(FakeService()), "tools/list")["result"]["tools"]]
        assert names == ["search", "overview", "list_documents", "get_document", "add_note", "add_document",
                         "index_status", "sync_project", "list_projects"]

    def test_call_dispatches_to_service(self):
        service = FakeService()
        resp = _call(srv.MCPServer(service), "tools/call", {"name": "search", "arguments": {"query": "odul", "top_k": 3}})
        assert resp["result"] == {"content": [{"type": "text", "text": "sonuc: odul"}], "isError": False}
        assert service.calls == [("search", "odul", None, 3)]

    def test_validation_error_is_tool_error_not_protocol_error(self):
        resp = _call(srv.MCPServer(FakeService()), "tools/call", {"name": "add_note", "arguments": {"title": "", "body": ""}})
        assert resp["result"]["isError"] is True
        assert "bos olamaz" in resp["result"]["content"][0]["text"]

    def test_unexpected_error_mentions_qdrant(self):
        resp = _call(srv.MCPServer(FakeService()), "tools/call", {"name": "list_projects", "arguments": {}})
        assert resp["result"]["isError"] is True and "Qdrant" in resp["result"]["content"][0]["text"]

    def test_unknown_argument_rejected(self):
        resp = _call(srv.MCPServer(FakeService()), "tools/call", {"name": "search", "arguments": {"query": "x", "rm": 1}})
        assert resp["result"]["isError"] is True


class TestServeLoop:
    def test_reads_lines_writes_one_json_per_response(self):
        stdin = io.StringIO(
            '{"jsonrpc":"2.0","id":1,"method":"ping"}\n'
            '{"jsonrpc":"2.0","method":"notifications/initialized"}\n'
            'bozuk json\n'
        )
        stdout = io.StringIO()
        srv.serve(srv.MCPServer(FakeService()), stdin, stdout)
        lines = [json.loads(l) for l in stdout.getvalue().splitlines()]
        assert lines[0] == {"jsonrpc": "2.0", "id": 1, "result": {}}
        assert lines[1]["error"]["code"] == -32700
        assert len(lines) == 2
