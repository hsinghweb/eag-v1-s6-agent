import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import asyncio
from google import genai
from concurrent.futures import TimeoutError
from functools import partial

# Load environment variables from .env file
load_dotenv()

# Access your API key and initialize Gemini client correctly
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

max_iterations = 10
last_response = None
iteration = 0
iteration_response = []
powerpoint_opened = False

async def generate_with_timeout(client, prompt, timeout=10):
    """Generate content with a timeout"""
    print("Starting LLM generation...")
    try:
        # Convert the synchronous generate_content call to run in a thread
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None, 
                lambda: client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
            ),
            timeout=timeout
        )
        print("LLM generation completed")
        return response
    except TimeoutError:
        print("LLM generation timed out!")
        raise
    except Exception as e:
        print(f"Error in LLM generation: {e}")
        raise

def reset_state():
    """Reset all global variables to their initial state"""
    global last_response, iteration, iteration_response, powerpoint_opened
    last_response = None
    iteration = 0
    iteration_response = []
    powerpoint_opened = False

async def main():
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            reset_state()  # Reset at the start of main
            print("Starting main execution...")
            
            # Create a single MCP server connection
            print("Establishing connection to MCP server...")
            server_params = StdioServerParameters(
                command="python",
                args=["mcp-server.py", "dev"]  # Add "dev" argument
            )

            async with stdio_client(server_params) as (read, write):
                print("Connection established, creating session...")
                async with ClientSession(read, write) as session:
                    print("Session created, initializing...")
                    try:
                        await session.initialize()
                    except Exception as e:
                        print(f"Failed to initialize session: {e}")
                        continue
                    
                    # Get available tools
                    print("Requesting tool list...")
                    try:
                        tools_result = await session.list_tools()
                        tools = tools_result.tools
                        print(f"Successfully retrieved {len(tools)} tools")
                    except Exception as e:
                        print(f"Failed to get tool list: {e}")
                        continue
                    
                    # Create system prompt with available tools
                    print("Creating system prompt...")
                    print(f"Number of tools: {len(tools)}")
                    
                    try:
                        tools_description = []
                        for i, tool in enumerate(tools):
                            try:
                                params = tool.inputSchema
                                desc = getattr(tool, 'description', 'No description available')
                                name = getattr(tool, 'name', f'tool_{i}')
                                
                                if 'properties' in params:
                                    param_details = []
                                    for param_name, param_info in params['properties'].items():
                                        param_type = param_info.get('type', 'unknown')
                                        param_details.append(f"{param_name}: {param_type}")
                                    params_str = ', '.join(param_details)
                                else:
                                    params_str = 'no parameters'

                                tool_desc = f"{i+1}. {name}({params_str}) - {desc}"
                                tools_description.append(tool_desc)
                                print(f"Added description for tool: {tool_desc}")
                            except Exception as e:
                                print(f"Error processing tool {i}: {e}")
                                tools_description.append(f"{i+1}. Error processing tool")
                        
                        tools_description = "\n".join(tools_description)
                        print("Successfully created tools description")
                    except Exception as e:
                        print(f"Error creating tools description: {e}")
                        tools_description = "Error loading tools"
                    
                    print("Created system prompt...")
                    
                    system_prompt = f"""You are a math agent that solves problems using structured, step-by-step reasoning and visualizes the results using PowerPoint. You must reason iteratively and explicitly separate calculation from visualization steps.

Available tools:
{tools_description}

Your workflow must strictly follow this structured loop for each problem:
1. Begin by identifying the necessary computations and perform **only** mathematical calculations first using FUNCTION_CALL.
2. Once calculations are complete, proceed to PowerPoint visualization:
   - Always begin with: POWERPOINT: open_powerpoint
   - Draw a rectangle to highlight results using coordinates (x1=2, y1=2, x2=7, y2=5).
   - Display the final computed value in this exact format:
     POWERPOINT: add_text_in_powerpoint|Final Result:\n<calculated_value>
   - End with: POWERPOINT: close_powerpoint

Always output in a **strict, one-line format** matching the following schemas:
- For function calls:  
  FUNCTION_CALL: function_name|param1|param2|...
- For PowerPoint operations:  
  POWERPOINT: operation|param1|param2|...
- For final results:  
  FINAL_ANSWER: [<computed_value>]

**Do not include any explanation or extra text.** Always use exactly one line per output.

You must also follow these constraints and practices:
- **Self-check**: If unsure of a value, re-calculate before moving to the next step.
- **Reasoning tags**: Internally categorize your reasoning type (e.g., arithmetic, logic).
- **Fallback behavior**: If a calculation or tool fails, return:  
  FINAL_ANSWER: [Error: Unable to compute]
- **Support for iterative use**: Always assume the next question might depend on prior context and computations.

Accepted array formats:
- Comma-separated: param1,param2,param3
- Bracketed list: [param1,param2,param3]

**Example outputs (use exactly these formats):**
- FUNCTION_CALL: add|5|3
- POWERPOINT: open_powerpoint
- POWERPOINT: add_text_in_powerpoint|Final Result:\n7.59982224609308e+33
- FINAL_ANSWER: [7.59982224609308e+33]
"""

                    query = """Find the ASCII values of characters in HIMANSHU and then return sum of exponentials of those values. 
                    Also, create a PowerPoint presentation showing the Final Answer inside a rectangle box."""
                    print("Starting iteration loop...")
                    
                    # Use global iteration variables
                    global iteration, last_response, powerpoint_opened
                    
                    while iteration < max_iterations:
                        print(f"\n--- Iteration {iteration + 1} ---")
                        if last_response is None:
                            current_query = query
                        else:
                            current_query = current_query + "\n\n" + " ".join(iteration_response)
                            current_query = current_query + "  What should I do next?"

                        # Get model's response with timeout
                        print("Preparing to generate LLM response...")
                        prompt = f"{system_prompt}\n\nQuery: {current_query}"
                        try:
                            response = await generate_with_timeout(client, prompt)
                            response_text = response.text.strip()
                            print(f"LLM Response: {response_text}")
                            
                            # Find the appropriate line in the response
                            for line in response_text.split('\n'):
                                line = line.strip()
                                if line.startswith(("FUNCTION_CALL:", "POWERPOINT:", "FINAL_ANSWER:")):
                                    response_text = line
                                    break
                            
                        except Exception as e:
                            print(f"Failed to get LLM response: {e}")
                            break

                        if response_text.startswith("FUNCTION_CALL:"):
                            _, function_info = response_text.split(":", 1)
                            parts = [p.strip() for p in function_info.split("|")]
                            func_name, params = parts[0], parts[1:]
                            
                            print(f"[Calling Tool] Raw function info: {function_info}")
                            print(f"[Calling Tool] Split parts: {parts}")
                            print(f"[Calling Tool] Function name: {func_name}")
                            print(f"[Calling Tool] Raw parameters: {params}")
                            
                            try:
                                # Find the matching tool to get its input schema
                                tool = next((t for t in tools if t.name == func_name), None)
                                if not tool:
                                    print(f"[Calling Tool] Available tools: {[t.name for t in tools]}")
                                    raise ValueError(f"Unknown tool: {func_name}")

                                print(f"[Calling Tool] Found tool: {tool.name}")
                                print(f"[Calling Tool] Tool schema: {tool.inputSchema}")

                                # Prepare arguments according to the tool's input schema
                                arguments = {}
                                schema_properties = tool.inputSchema.get('properties', {})
                                print(f"[Calling Tool] Schema properties: {schema_properties}")

                                for param_name, param_info in schema_properties.items():
                                    if not params:  # Check if we have enough parameters
                                        raise ValueError(f"Not enough parameters provided for {func_name}")
                                        
                                    value = params.pop(0)  # Get and remove the first parameter
                                    param_type = param_info.get('type', 'string')
                                    
                                    print(f"[Calling Tool] Converting parameter {param_name} with value {value} to type {param_type}")
                                    
                                    # Convert the value to the correct type based on the schema
                                    if param_type == 'integer':
                                        arguments[param_name] = int(value)
                                    elif param_type == 'number':
                                        arguments[param_name] = float(value)
                                    elif param_type == 'array':
                                        # Handle array input - if it's already a string representation of a list
                                        if value.startswith('[') and value.endswith(']'):
                                            # Parse the array string properly
                                            array_str = value.strip('[]')
                                            if array_str:
                                                arguments[param_name] = [int(x.strip()) for x in array_str.split(',')]
                                            else:
                                                arguments[param_name] = []
                                        else:
                                            # If it's a comma-separated string without brackets
                                            if ',' in value:
                                                arguments[param_name] = [int(x.strip()) for x in value.split(',')]
                                            else:
                                                # If it's a single value, make it a single-item list
                                                arguments[param_name] = [int(value)]
                                    else:
                                        arguments[param_name] = str(value)

                                print(f"[Calling Tool] Final arguments: {arguments}")
                                print(f"[Calling Tool] Calling tool {func_name}")
                                
                                result = await session.call_tool(func_name, arguments=arguments)
                                print(f"[Calling LLM] Raw result: {result}")
                                
                                # Get the full result content
                                if hasattr(result, 'content'):
                                    print(f"[Calling LLM] Result has content attribute")
                                    # Handle multiple content items
                                    if isinstance(result.content, list):
                                        iteration_result = [
                                            item.text if hasattr(item, 'text') else str(item)
                                            for item in result.content
                                        ]
                                    else:
                                        iteration_result = str(result.content)
                                else:
                                    print(f"[Calling LLM] Result has no content attribute")
                                    iteration_result = str(result)
                                    
                                print(f"[Calling LLM] Final iteration result: {iteration_result}")
                                
                                # Format the response based on result type
                                if isinstance(iteration_result, list):
                                    result_str = f"[{', '.join(iteration_result)}]"
                                else:
                                    result_str = str(iteration_result)
                                
                                iteration_response.append(
                                    f"In the {iteration + 1} iteration you called {func_name} with {arguments} parameters, "
                                    f"and the function returned {result_str}."
                                )
                                last_response = iteration_result

                            except Exception as e:
                                print(f"[Calling LLM] Error details: {str(e)}")
                                print(f"[Calling LLM] Error type: {type(e)}")
                                import traceback
                                traceback.print_exc()
                                iteration_response.append(f"Error in iteration {iteration + 1}: {str(e)}")
                                break

                        elif response_text.startswith("POWERPOINT:"):
                            _, operation_info = response_text.split(":", 1)
                            parts = [p.strip() for p in operation_info.split("|")]
                            operation, params = parts[0], parts[1:]
                            
                            print(f"[Calling Tool] PowerPoint operation: {operation}")
                            print(f"[Calling Tool] PowerPoint parameters: {params}")
                            
                            try:
                                if operation == "open_powerpoint":
                                    if not powerpoint_opened:
                                        result = await session.call_tool("open_powerpoint")
                                        powerpoint_opened = True
                                    else:
                                        iteration_response.append("PowerPoint is already open")
                                        continue
                                elif operation == "draw_rectangle":
                                    if powerpoint_opened:
                                        # Convert parameters to integers before passing
                                        try:
                                            x1, y1, x2, y2 = map(int, params)
                                            result = await session.call_tool(
                                                "draw_rectangle",
                                                arguments={
                                                    "x1": x1,
                                                    "y1": y1,
                                                    "x2": x2,
                                                    "y2": y2
                                                }
                                            )
                                        except (ValueError, TypeError) as e:
                                            print(f"[Calling Tool] Error converting rectangle parameters: {e}")
                                            print(f"[Calling Tool] Raw parameters: {params}")
                                            iteration_response.append(f"Error: Invalid rectangle parameters - {str(e)}")
                                            continue
                                    else:
                                        iteration_response.append("PowerPoint must be opened first")
                                        continue
                                elif operation == "add_text_in_powerpoint":
                                    if powerpoint_opened:
                                        # Get the full text after the operation name
                                        full_text = response_text.split("|", 1)[1].strip()
                                        # Remove any extra quotes and handle newlines
                                        full_text = full_text.replace('"', '').replace("\\n", "\n")
                                        
                                        # If this is the final result text, append the calculated value
                                        if "Final Result:" in full_text:
                                            # Find the last calculation result from iteration_response
                                            calc_result = next((resp.split("returned")[1].strip() 
                                                for resp in reversed(iteration_response) 
                                                if "int_list_to_exponential_sum" in resp), None)
                                            if calc_result:
                                                full_text = f"Final Result:\n{calc_result}"
                                        
                                        # Ensure proper newline handling
                                        full_text = full_text.replace('\n\n', '\n')
                                        print(f"[Calling Tool] Full text to add: {repr(full_text)}")  # Show raw string representation
                                        print(f"[Calling Tool] Text length: {len(full_text)}")
                                        print(f"[Calling Tool] Text contains newlines: {'\\n' in full_text}")
                                        
                                        # Split the text into lines and rejoin with proper newlines
                                        lines = full_text.split('\n')
                                        formatted_text = '\n'.join(line.strip() for line in lines if line.strip())
                                        print(f"[Calling Tool] Formatted text: {repr(formatted_text)}")
                                        
                                        result = await session.call_tool(
                                            "add_text_in_powerpoint",
                                            arguments={
                                                "text": formatted_text
                                            }
                                        )
                                    else:
                                        iteration_response.append("PowerPoint must be opened first")
                                        continue
                                elif operation == "close_powerpoint":
                                    if powerpoint_opened:
                                        result = await session.call_tool("close_powerpoint")
                                        powerpoint_opened = False
                                        # Add final answer here and break the loop
                                        print("\n=== Agent Execution Complete ===")
                                        final_answer = next(resp.split("returned")[1].strip() 
                                            for resp in iteration_response 
                                            if "int_list_to_exponential_sum" in resp)
                                        print(f"Final Answer: {final_answer}")
                                        break
                                    else:
                                        iteration_response.append("PowerPoint is already closed")
                                        continue
                                else:
                                    raise ValueError(f"Unknown PowerPoint operation: {operation}")
                                
                                print(f"[MCP Tool → LLM] PowerPoint result: {result}")
                                iteration_response.append(f"PowerPoint operation '{operation}' completed successfully.")
                                
                            except Exception as e:
                                print(f"[MCP Tool → LLM] Error in PowerPoint operation: {str(e)}")
                                iteration_response.append(f"Error in PowerPoint operation: {str(e)}")
                                break

                        elif response_text.startswith("FINAL_ANSWER:"):
                            # Skip this section since we're handling final answer after close_powerpoint
                            continue

                        iteration += 1

            break  # If we get here, everything worked fine
            
        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected, cleaning up...")
            reset_state()
            break
        except Exception as e:
            print(f"Error in main execution (attempt {retry_count + 1}/{max_retries}): {e}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying in 5 seconds...")
                await asyncio.sleep(5)
            else:
                print("Max retries reached, exiting...")
                raise
        finally:
            reset_state()  # Reset at the end of main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting due to keyboard interrupt...")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    
    
