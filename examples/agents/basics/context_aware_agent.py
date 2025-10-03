"""
Usage examples for ContextAwareChatToolAgent
"""
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.provider import OpenAIChatAPI  # or your provider
from ToolAgents import ToolRegistry, FunctionTool
from ToolAgents.agent_context.agent_context import AgentContext
from ToolAgents.agent_context.context_aware_agent import ContextAwareChatToolAgent


# Example 1: Basic usage with context
def example_basic_conversation():
    """Simple multi-turn conversation with context."""

    # Initialize with context
    context = AgentContext(max_buffer_size=100)
    context.set_system_prompt("You are a helpful Python programming assistant.")

    agent = ContextAwareChatToolAgent(
        chat_api=OpenAIChatAPI(api_key="your-key", base_url="http://127.0.0.1:8080/v1", model="george"),
        context=context
    )

    # Simple chat interface - context is automatically managed
    response1 = agent.chat("What's the difference between a list and a tuple?")
    print(response1)

    # Context remembers previous conversation
    response2 = agent.chat("Can you show me an example?")
    print(response2)

    # Check context size
    print(f"Context size: {agent.get_context_size()} messages")
    print(f"Estimated tokens: {agent.get_context_token_estimate()}")


# Example 2: Using with tools
def example_with_tools():
    """Using context-aware agent with function tools."""

    def calculate(operation: str, a: float, b: float) -> float:
        """
        Perform basic math operations.

        Args:
            operation (str): operation to perform
            a (float): first parameter
            b (float): second parameter
        """
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero"
        }
        return operations.get(operation, lambda x, y: "Unknown operation")(a, b)

    # Setup
    context = AgentContext()
    context.set_system_prompt("You are a math assistant. Use the calculate tool when needed.")

    agent = ContextAwareChatToolAgent(
        chat_api=OpenAIChatAPI(api_key="your-key", base_url="http://127.0.0.1:8080/v1", model="george"),
        context=context
    )

    registry = ToolRegistry()
    registry.add_tool(FunctionTool(calculate))

    # Conversation with tool usage
    response = agent.get_response(
        [ChatMessage.create_user_message("What's 15 * 23?")],
        tool_registry=registry,
        use_context=True
    )
    print(response.response)


# Example 3: Branching conversations
def example_branching():
    """Explore different conversation paths."""

    context = AgentContext()
    context.set_system_prompt("You are a creative writing assistant.")

    agent = ContextAwareChatToolAgent(
        chat_api=OpenAIChatAPI(api_key="your-key", base_url="http://127.0.0.1:8080/v1", model="george"),
        context=context
    )

    # Start conversation
    agent.chat("Help me write a story about a robot")
    agent.chat("The robot should be friendly")

    # Branch 1: Comedy path
    comedy_agent = agent.fork()
    comedy_response = comedy_agent.chat("Make it a comedy story")

    # Branch 2: Drama path (original agent)
    drama_response = agent.chat("Make it a dramatic story")

    print("Comedy branch:", comedy_response)
    print("Drama branch:", drama_response)


# Example 4: Context management
def example_context_management():
    """Managing context size and compression."""

    context = AgentContext(max_buffer_size=10)  # Small buffer for demo

    agent = ContextAwareChatToolAgent(
        chat_api=OpenAIChatAPI(api_key="your-key", base_url="http://127.0.0.1:8080/v1", model="george"),
        context=context,
        log_output=True  # Enable logging
    )

    # Have a long conversation
    topics = ["Python", "JavaScript", "Rust", "Go", "TypeScript"]
    for topic in topics:
        agent.chat(f"Tell me about {topic}")
        print(f"Context size after {topic}: {agent.get_context_size()}")

    # Context automatically trimmed to max_buffer_size

    # Manual compression if needed
    if agent.get_context_token_estimate() > 2000:
        agent.compress_context(target_size=5)
        print(f"Compressed to {agent.get_context_size()} messages")

    # Reset for new conversation
    agent.reset_context(preserve_system=True)  # Keep system prompt
    agent.chat("Let's talk about something new")


# Example 5: Direct usage (backwards compatible)
def example_backwards_compatible():
    """Using without context (standard ChatToolAgent behavior)."""

    agent = ContextAwareChatToolAgent(
        chat_api=OpenAIChatAPI(api_key="your-key", base_url="http://127.0.0.1:8080/v1", model="george")
        # No context provided - creates empty one
    )

    # Use like regular ChatToolAgent
    messages = [
        ChatMessage.create_system_message("You are helpful."),
        ChatMessage.create_user_message("Hello!")
    ]

    # Disable context with use_context=False
    response = agent.get_response(
        messages,
        use_context=False  # Behaves exactly like ChatToolAgent
    )
    print(response.response)

    # Or enable context management
    agent.set_system_prompt("You are a helpful assistant.")
    response = agent.get_response(
        [ChatMessage.create_user_message("Hello!")],
        use_context=True  # Uses context
    )
    print(response.response)


# Example 6: Async usage
async def example_async():
    """Async context-aware agent."""
    from ToolAgents.provider.chat_api_provider.open_ai import AsyncOpenAIChatAPI
    from ToolAgents.agent_context.context_aware_agent import AsyncContextAwareChatToolAgent

    context = AgentContext()
    context.set_system_prompt("You are an async assistant.")

    agent = AsyncContextAwareChatToolAgent(
        chat_api=AsyncOpenAIChatAPI(api_key="your-key", base_url="http://127.0.0.1:8080/v1", model="george"),
        context=context
    )

    # Async chat
    response = await agent.chat("Tell me about async programming")
    print(response)

    # Context is maintained across async calls
    response = await agent.chat("What are the main benefits?")
    print(response)


# Example 7: Conversation analysis
def example_conversation_analysis():
    """Analyze conversation patterns."""

    context = AgentContext()
    agent = ContextAwareChatToolAgent(
        chat_api=OpenAIChatAPI(api_key="your-key", base_url="http://127.0.0.1:8080/v1", model="george"),
        context=context
    )

    # Have a conversation
    agent.chat("What's machine learning?")
    agent.chat("How does neural network training work?")
    agent.chat("What about overfitting?")

    # Analyze conversation
    last_user = context.find_last_user_message()
    print(f"Last user question: {last_user.get_as_text()}")

    pairs = context.get_conversation_pairs()
    print(f"Total Q&A pairs: {len(pairs)}")

    # Get conversation summary
    for user_msg, assistant_msg in pairs:
        print(f"Q: {user_msg.get_as_text()[:50]}...")
        if assistant_msg:
            print(f"A: {assistant_msg.get_as_text()[:50]}...")


if __name__ == "__main__":
    # Run examples
    example_basic_conversation()
    example_with_tools()
    example_branching()
    example_context_management()
    example_backwards_compatible()
    example_conversation_analysis()

    # For async example
    # import asyncio
    # asyncio.run(example_async())