import json

from fastapi import FastAPI, Request
import os

from dotenv import load_dotenv
from starlette.responses import HTMLResponse, JSONResponse

from ToolAgents import ToolRegistry
from ToolAgents.provider import AnthropicChatAPI, AnthropicSettings
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import ChatHistory

from ToolAgents.messages import ChatMessage

load_dotenv()

provider = AnthropicChatAPI(os.getenv("ANTHROPIC_API_KEY"), "claude-3-5-sonnet-20241022")
agent = ChatToolAgent(chat_api=provider, debug_output=False)
app = FastAPI()

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def catch_all(request: Request, path: str):
    message = ChatMessage.create_user_message(f"""Your task is to generate web content, like JSON data or complete websites based on request strings to an endpoint. You should deliver a realistic web content. Only write the web content like JSON data without additional commentary.

<request_string>
{path}
</request_string>""")
    output = agent.get_response([message])
    print(output.response)
    try:
        data = json.loads(output.response)
        return JSONResponse(content=data)
    except json.decoder.JSONDecodeError:
        return HTMLResponse(content=output.response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)