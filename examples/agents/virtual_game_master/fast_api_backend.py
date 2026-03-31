import dataclasses

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import Dict
import json
import os
import asyncio
from contextlib import asynccontextmanager

from virtual_game_master import VirtualGameMasterConfig, VirtualGameMaster
from chat_api_selector import VirtualGameMasterChatAPISelector


class ConfigUpdate(BaseModel):
    GAME_SAVE_FOLDER: str
    INITIAL_GAME_STATE: str
    TEMPERATURE: float
    TOP_P: float
    TOP_K: int
    MIN_P: float
    TFS_Z: float

    def to_dict(self):
        return {
            "GAME_SAVE_FOLDER": self.GAME_SAVE_FOLDER,
            "INITIAL_GAME_STATE": self.INITIAL_GAME_STATE,
            "TEMPERATURE": self.TEMPERATURE,
            "TOP_P": self.TOP_P,
            "TOP_K": self.TOP_K,
            "MIN_P": self.MIN_P,
            "TFS_Z": self.TFS_Z,
        }

class Message(BaseModel):
    content: str


class EditMessage(BaseModel):
    id: int
    content: str


class TemplateFields(BaseModel):
    fields: Dict[str, str]


@dataclasses.dataclass
class State:
    rpg_app: VirtualGameMaster


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    config = VirtualGameMasterConfig.from_env()
    api_selector = VirtualGameMasterChatAPISelector(config)
    api = api_selector.get_api()
    app.state = State(rpg_app=VirtualGameMaster(config, api, True))
    app.state.rpg_app.load()
    yield
    # Shutdown
    # Add any cleanup code here if needed


app = FastAPI(lifespan=lifespan)
app.state = None

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.post("/api/send_message")
async def send_message(message: Message):
    response_generator, should_exit = app.state.rpg_app.process_input(message.content, stream=True)
    response = "".join(list(response_generator))
    return {"response": response, "should_exit": should_exit}


@app.post("/api/edit_message")
async def edit_message(edit_message: EditMessage):
    success = app.state.rpg_app.edit_message(edit_message.id, edit_message.content)
    app.state.rpg_app.save()
    if success:
        return {"status": "success"}
    else:
        raise HTTPException(status_code=404, detail="Message not found")


@app.get("/api/get_template_fields")
async def get_template_fields():
    return {"fields": app.state.rpg_app.game_state.template_fields}


@app.post("/api/update_template_fields")
async def update_template_fields(fields: TemplateFields):
    app.state.rpg_app.game_state.template_fields.update(fields.fields)
    app.state.rpg_app.save()
    return {"status": "success"}


@app.post("/api/save_game")
async def save_game():
    app.state.rpg_app.save()
    return {"status": "success"}


@app.get("/api/get_chat_history")
async def get_chat_history():
    return {"history": app.state.rpg_app.history.to_list(), "next_message_id": app.state.rpg_app.next_message_id}


@app.delete("/api/delete_message/{msg_id}")
async def get_delete_message(msg_id: int):
    result = app.state.rpg_app.history.delete_message(msg_id)
    app.state.rpg_app.save()
    if result:
        return {"status": "success", "next_message_id": app.state.rpg_app.next_message_id}
    else:
        raise HTTPException(status_code=404, detail="Message not found")


@app.get("/api/get_config")
async def get_config():
    return {
        "GAME_SAVE_FOLDER": app.state.rpg_app.config.GAME_SAVE_FOLDER,
        "INITIAL_GAME_STATE": app.state.rpg_app.config.INITIAL_GAME_STATE,
        "TEMPERATURE": app.state.rpg_app.config.TEMPERATURE,
        "TOP_P": app.state.rpg_app.config.TOP_P,
        "TOP_K": app.state.rpg_app.config.TOP_K,
        "MIN_P": app.state.rpg_app.config.MIN_P,
        "TFS_Z": app.state.rpg_app.config.TFS_Z,
    }


@app.post("/api/update_config")
async def update_config(config_update: ConfigUpdate):
    try:

        app.state.rpg_app.config.update(config_update.to_dict())
        config = app.state.rpg_app.config
        api_selector = VirtualGameMasterChatAPISelector(config)
        api = api_selector.get_api()
        app.state = State(rpg_app=VirtualGameMaster(config, api))
        app.state.rpg_app.load()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/save_config")
async def save_config(config_update: ConfigUpdate):
    try:

        app.state.rpg_app.config.update(config_update.to_dict())
        app.state.rpg_app.config.to_env()

        config = VirtualGameMasterConfig.from_env()
        api_selector = VirtualGameMasterChatAPISelector(config)
        api = api_selector.get_api()
        app.state = State(rpg_app=VirtualGameMaster(config, api))
        app.state.rpg_app.load()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/get_chat_history_folders")
async def get_chat_history_folders():
    chat_history_path = os.path.join(os.path.dirname(__file__), "chat_history")
    folders = [os.path.join("chat_history", f) for f in os.listdir(chat_history_path) if os.path.isdir(os.path.join(chat_history_path, f))]
    return {"folders": folders, "active": os.path.join("chat_history", os.path.basename(app.state.rpg_app.config.GAME_SAVE_FOLDER))}


@app.get("/api/get_game_starters")
async def get_game_starters():
    game_starters_path = os.path.join(os.path.dirname(__file__), "game_starters")
    starters = [os.path.join("game_starters", f) for f in os.listdir(game_starters_path) if f.endswith(".yaml")]
    return {"game_starters": starters, "active": os.path.join("game_starters", os.path.basename(app.state.rpg_app.config.INITIAL_GAME_STATE))}


@app.post("/api/create_game_save_folder/{folder_name}")
async def create_game_save_folder(folder_name: str):
    try:
        chat_history_path = os.path.join(os.path.dirname(__file__), "chat_history")
        new_folder_path = os.path.join(chat_history_path, folder_name)

        if os.path.exists(new_folder_path):
            raise HTTPException(status_code=400, detail="Folder already exists")

        os.makedirs(new_folder_path)
        return {"status": "success", "message": f"Folder '{folder_name}' created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            response_generator, should_exit = app.state.rpg_app.process_input(message['content'], stream=True)

            async for chunk in async_generator(response_generator):
                await websocket.send_text(json.dumps({"type": "chunk", "content": chunk}))
                await asyncio.sleep(0)  # Allow other tasks to run

            await websocket.send_text(json.dumps(
                {"type": "end", "should_exit": should_exit, "next_message_id": app.state.rpg_app.next_message_id}))

            if should_exit:
                break
    except WebSocketDisconnect:
        print("WebSocket disconnected")


async def async_generator(sync_generator):
    for item in sync_generator:
        yield item
        await asyncio.sleep(0)  # Allow other tasks to run


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
