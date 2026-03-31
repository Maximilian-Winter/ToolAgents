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


# Usage
if __name__ == "__main__":
    config = VirtualGameMasterConfig.from_env(".env")
    config.GAME_SAVE_FOLDER = "chat_history/new_april_new_26"
    api_selector = VirtualGameMasterChatAPISelector(config)
    api = api_selector.get_api()
    vgm_app = VirtualGameMaster(config, api, True)
    run_cli(vgm_app)
