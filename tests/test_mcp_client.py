from services.mcp_client import StdioMCPClient


def test_stdio_client_initializes_and_lists_tools():
    with StdioMCPClient(request_timeout=30) as client:
        info = client.initialize()
        tools = client.list_tools()

    assert info["serverInfo"]["name"] == "rag-knowledge-base-mcp"
    names = {tool["name"] for tool in tools}
    assert "search_knowledge_base" in names
    assert "list_knowledge_documents" in names
