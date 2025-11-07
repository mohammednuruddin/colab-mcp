from fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("runtime")


# Add an execute code tool
@mcp.tool()
def execute_code(code: str):
    """(Eventually) Evaluates code in a Colab kernel."""
    # But for now, just evals here.
    return eval(code)
