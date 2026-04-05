import sys

from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent

from virtual_game_master import VirtualGameMasterConfig, VirtualGameMaster
from chat_api_selector import VirtualGameMasterChatAPISelector


def display_recent_messages(app: VirtualGameMaster, num_messages: int = 4):
    recent_messages = app.history.get_last_k_messages(app.chat_id, num_messages)
    for message in recent_messages:
        role = "Game Master" if message.role == "assistant" else "You"
        print(f"{role}: {message.content}\n")


def run_cli(app: VirtualGameMaster):
    app.load()
    print("Welcome to the Virtual Game Master App! Type in your next message to the Game Master or use '" + app.config.COMMAND_PREFIX + "help' to show all available commands.")
    print("Use '+' at the end of a line to continue input on the next line.")

    display_recent_messages(app)
    while True:
        user_input = ""
        while True:
            line = input("You: " if not user_input else "... ")
            if line.endswith('+'):
                user_input += line[:-1] + "\n"
            else:
                user_input += line
                break

        response_generator, should_exit = app.process_input(user_input.strip(), True)

        if should_exit:
            break

        print(f"\n", flush=True)
        if isinstance(response_generator, str):
            print(f"Game Master: {response_generator}\n")
        else:
            print(f"Game Master:", end="", flush=True)
            for tok in response_generator:
                print(tok, end="", flush=True)
            print("\n")


def setup_gm_tools(
    base_url: str = "http://localhost:8000",
    campaign_id: int = 1,
    scenario_path: str | None = None,
    import_lore: bool = False,
) -> ToolRegistry | None:
    """
    Set up GM Tool integration. Returns a ToolRegistry with the 2 read
    tools, or None if the server is unavailable.
    """
    try:
        from gm_tools import GMToolkit
    except ImportError:
        print("  [GM Tools] gm_tools module not found.")
        return None

    toolkit = GMToolkit(base_url=base_url, campaign_id=campaign_id)
    if not toolkit.is_available():
        print(f"  [GM Tools] Server at {base_url} not reachable.")
        print(f"  [GM Tools] Start it with: cd gm_tool && uvicorn app:app --port 8000")
        toolkit.close()
        return None

    print(f"  [GM Tools] Connected to {base_url}, campaign #{campaign_id}")

    if import_lore and scenario_path:
        print(f"  [GM Tools] Importing world lore from {scenario_path}...")
        toolkit.import_world_lore_from_yaml(scenario_path)

    registry = ToolRegistry()
    registry.add_tools(toolkit.get_read_tools())
    print(f"  [GM Tools] 2 retrieval tools registered")
    return registry


# Usage
if __name__ == "__main__":
    config = VirtualGameMasterConfig.from_env(".env")
    config.GAME_SAVE_FOLDER = "chat_history/new_april_new_26"
    api_selector = VirtualGameMasterChatAPISelector(config)
    api = api_selector.get_api()

    # -- GM Tool integration (optional) --
    # --gm-tools       : enable campaign database retrieval tools
    # --import-lore    : import world lore from the scenario YAML
    tool_registry = None
    if "--gm-tools" in sys.argv:
        tool_registry = setup_gm_tools(
            base_url="http://localhost:8000",
            campaign_id=1,
            scenario_path=config.INITIAL_GAME_STATE,
            import_lore="--import-lore" in sys.argv,
        )

    tool_agent = ChatToolAgent(chat_api=api)
    vgm_app = VirtualGameMaster(
        config=config,
        tool_agent=tool_agent,
        debug_mode=True,
        tool_registry=tool_registry,
    )
    run_cli(vgm_app)
