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
                    
                    system_prompt = """You are a math agent that solves problems using structured, step-by-step reasoning and visualizes the results using PowerPoint. You must reason iteratively and explicitly separate calculation from visualization steps.

Available tools:
{tools_description}

Your workflow must strictly follow this structured loop for each problem:
1. Begin by identifying the necessary computations and perform **only** mathematical calculations first using a function call in JSON format:
   - For ASCII values, use 'strings_to_chars_to_int'
   - For exponential sums, use 'int_list_to_exponential_sum'
2. Once calculations are complete, proceed to PowerPoint visualization in JSON format:
   - Begin with PowerPoint open operation
   - Draw a rectangle to highlight results using coordinates (x1=2, y1=2, x2=7, y2=5)
   - Display the final computed value
   - End with PowerPoint close operation

All outputs MUST be in valid JSON format following these schemas:

1. For function calls:
{
  "type": "function_call",
  "function": "function_name",
  "params": {"param1": "value1", "param2": "value2"}
}

2. For PowerPoint operations:
{
  "type": "powerpoint",
  "operation": "operation_name",
  "params": {"param1": "value1", "param2": "value2"}
}

3. For final results:
{
  "type": "final_answer",
  "value": "computed_value"
}

Constraints and practices:
- **Self-check**: If unsure of a value, re-calculate before moving to the next step
- **Reasoning tags**: Internally categorize your reasoning type (e.g., arithmetic, logic)
- **Fallback behavior**: If a calculation or tool fails, return: {"type": "final_answer", "value": "Error: Unable to compute"}
- **Support for iterative use**: Always assume the next question might depend on prior context and computations

Accepted array formats:
- Comma-separated: param1,param2,param3
- Bracketed list: [param1,param2,param3]

**Example outputs (use exactly these formats):**
{
  "type": "function_call",
  "function": "strings_to_chars_to_int",
  "params": {"string": "HIMANSHU"}
}
{
  "type": "powerpoint",
  "operation": "open_powerpoint",
  "params": {"dummy": null}
}
{
  "type": "powerpoint",
  "operation": "add_text_in_powerpoint",
  "params": {"text": "Final Result:\n7.59982224609308e+33"}
}
{
  "type": "final_answer",
  "value": 7.59982224609308e+33
}"""

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
                            # Remove markdown code block formatting if present
                            response_text = response.text.strip()
                            # Remove markdown formatting and clean up JSON string
                            if '```' in response_text:
                                response_text = response_text.split('```')[1] if len(response_text.split('```')) > 1 else response_text
                            response_text = response_text.replace('json\n', '').strip()
                            # Remove any leading/trailing whitespace or quotes
                            response_text = response_text.strip('`').strip('"').strip()
                            print(f"LLM Response: {response_text}")
                            
                            # Parse the JSON response
                            import json
                            try:
                                # Try to parse the cleaned response text
                                try:
                                    response_json = json.loads(response_text)
                                except json.JSONDecodeError:
                                    # If initial parse fails, try to extract JSON from the text
                                    import re
                                    json_match = re.search(r'\{[^}]+\}', response_text)
                                    if json_match:
                                        response_text = json_match.group(0)
                                        response_json = json.loads(response_text)
                                    else:
                                        raise ValueError("No valid JSON found in response")
                                
                                if not isinstance(response_json, dict) or 'type' not in response_json:
                                    raise ValueError("Invalid response format")
                                
                                # Validate response against expected schemas
                                valid_types = ['function_call', 'powerpoint', 'final_answer']
                                if response_json['type'] not in valid_types:
                                    raise ValueError(f"Invalid response type. Expected one of {valid_types}")
                                
                                if response_json['type'] == 'function_call':
                                    if 'function' not in response_json or 'params' not in response_json:
                                        raise ValueError("Invalid function_call format")
                                elif response_json['type'] == 'powerpoint':
                                    if 'operation' not in response_json or 'params' not in response_json:
                                        raise ValueError("Invalid powerpoint operation format")
                                elif response_json['type'] == 'final_answer':
                                    if 'value' not in response_json:
                                        raise ValueError("Invalid final_answer format")
                            except json.JSONDecodeError as e:
                                print(f"Failed to parse JSON response: {e}")
                                break
                            
                        except Exception as e:
                            print(f"Failed to get LLM response: {e}")
                            break

                        if response_json['type'] == 'function_call':
                            func_name = response_json['function']
                            params = response_json['params']
                            
                            print(f"[Calling Tool] Function name: {func_name}")
                            print(f"[Calling Tool] Parameters: {params}")
                            
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
                                    if param_name not in params:  # Check if parameter is provided
                                        if param_name in tool.inputSchema.get('required', []):
                                            raise ValueError(f"Required parameter {param_name} not provided for {func_name}")
                                        continue
                                        
                                    value = params[param_name]
                                    param_type = param_info.get('type', 'string')
                                    
                                    print(f"[Calling Tool] Converting parameter {param_name} with value {value} to type {param_type}")
                                    
                                    # Convert the value to the correct type based on the schema
                                    if param_type == 'integer':
                                        arguments[param_name] = int(value)
                                    elif param_type == 'number':
                                        arguments[param_name] = float(value)
                                    elif param_type == 'array':
                                        if isinstance(value, list):
                                            arguments[param_name] = value
                                        elif isinstance(value, str):
                                            # Handle string representation of array
                                            if value.startswith('[') and value.endswith(']'):
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
                                            raise ValueError(f"Invalid array format for parameter {param_name}")
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

                        elif response_json['type'] == 'powerpoint':
                            operation = response_json['operation']
                            params = response_json['params']
                            
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
                                        try:
                                            result = await session.call_tool(
                                                "draw_rectangle",
                                                arguments=params
                                            )
                                        except Exception as e:
                                            print(f"[Calling Tool] Error with rectangle parameters: {e}")
                                            iteration_response.append(f"Error: Invalid rectangle parameters - {str(e)}")
                                            continue
                                    else:
                                        iteration_response.append("PowerPoint must be opened first")
                                        continue
                                elif operation == "add_text_in_powerpoint":
                                    if powerpoint_opened:
                                        text = params.get('text', '')
                                        # Handle newlines in JSON string
                                        
                                        # If this is the final result text, append the calculated value
                                        if "Final Result:" in text:
                                            # Find the last calculation result from iteration_response
                                            calc_result = next((resp.split("returned")[1].strip() 
                                                for resp in reversed(iteration_response) 
                                                if "returned" in resp), None)
                                            if calc_result:
                                                text = f"Final Result:\n{calc_result}"
                                        
                                        result = await session.call_tool(
                                            "add_text_in_powerpoint",
                                            arguments={"text": text}
                                        )
                                    else:
                                        iteration_response.append("PowerPoint must be opened first")
                                        continue
                                elif operation == "close_powerpoint":
                                    if powerpoint_opened:
                                        result = await session.call_tool("close_powerpoint")
                                        powerpoint_opened = False
                                    else:
                                        iteration_response.append("PowerPoint is not open")
                                        continue
                                else:
                                    iteration_response.append(f"Unknown PowerPoint operation: {operation}")
                                    continue
                                
                                # Get the full result content
                                if hasattr(result, 'content'):
                                    # Handle multiple content items
                                    if isinstance(result.content, list):
                                        iteration_result = [
                                            item.text if hasattr(item, 'text') else str(item)
                                            for item in result.content
                                        ]
                                    else:
                                        iteration_result = str(result.content)
                                else:
                                    iteration_result = str(result)
                                    
                                # Format the response based on result type
                                if isinstance(iteration_result, list):
                                    result_str = f"[{', '.join(iteration_result)}]"
                                else:
                                    result_str = str(iteration_result)
                                
                                iteration_response.append(
                                    f"In the {iteration + 1} iteration you performed PowerPoint operation {operation} "
                                    f"with {params} parameters, and the operation returned {result_str}."
                                )
                                last_response = iteration_result
                                
                            except Exception as e:
                                print(f"Error in PowerPoint operation: {e}")
                                iteration_response.append(f"Error in PowerPoint operation: {str(e)}")
                                break
                                
                        elif response_json['type'] == 'final_answer':
                            value = response_json['value']
                            iteration_response.append(f"Final answer: {value}")
                            break
                            
                        iteration += 1
                        
                    if iteration >= max_iterations:
                        print("Reached maximum iterations")
                        break
                        
                    print("\nFinal Results:")
                    for resp in iteration_response:
                        print(resp)
                        
                    return
                    
        except Exception as e:
            print(f"Error in main loop: {e}")
            retry_count += 1
            if retry_count >= max_retries:
                print("Maximum retries reached")
                break
            print(f"Retrying... ({retry_count}/{max_retries})")
            continue

if __name__ == "__main__":
    asyncio.run(main())
    
    
