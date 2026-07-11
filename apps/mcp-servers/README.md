# MCP Servers

This package contains dependency-free MCP-facing handlers and stdio JSON-RPC
servers for the Dullahan CAL and EDL tools. The handlers accept MCP-style
argument dictionaries, validate them against the same Pydantic contracts used
by the services, and call the HTTP adapters.

The YAML manifests under `mcp/` describe the intended tool names and schemas.
Run the stdio servers with:

```bash
dullahan-mcp-cal
dullahan-mcp-edl
```

The servers expose `send_to_CAL` and `send_to_EDL` through `tools/list` and
`tools/call`. Configure their HTTP targets with `CAL_BASE_URL` and
`EDL_BASE_URL`.
