import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # Create MCP server connection
    server_params = StdioServerParameters(
        command="python",
        args=["mcp-server.py", "dev"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize session
            await session.initialize()
            
            # Step 1: Convert HIMANSHU to ASCII values
            ascii_result = await session.call_tool(
                "strings_to_chars_to_int",
                arguments={"string": "HIMANSHU"}
            )
            # Extract ASCII values from response
            ascii_values = [int(content.text) for content in ascii_result.content]
            print(f"ASCII values: {ascii_values}")
            
            # Step 2: Calculate sum of exponentials
            exp_result = await session.call_tool(
                "int_list_to_exponential_sum",
                arguments={"int_list": ascii_values}
            )
            # Extract exponential sum from response
            exp_sum = float(exp_result.content[0].text)
            print(f"Sum of exponentials: {exp_sum}")
            
            # Step 3: Create PowerPoint presentation
            # Open PowerPoint
            await session.call_tool("open_powerpoint")
            
            # Draw rectangle
            await session.call_tool(
                "draw_rectangle",
                arguments={"x1": 2, "y1": 2, "x2": 7, "y2": 5}
            )
            
            # Add text with result
            await session.call_tool(
                "add_text_in_powerpoint",
                arguments={"text": f"Final Result:\n{exp_sum}"}
            )
            
            # Close PowerPoint
            await session.call_tool("close_powerpoint")

if __name__ == "__main__":
    asyncio.run(main())