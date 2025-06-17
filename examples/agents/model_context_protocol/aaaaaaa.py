import asyncio
import sys
import threading
import queue
import json
from abc import ABC, abstractmethod
from typing import Optional, AsyncGenerator
import platform


# Assuming these are defined elsewhere in your code
class MCPMessage:
    def __init__(self, jsonrpc="2.0", id=None, method=None, params=None, result=None, error=None):
        self.jsonrpc = jsonrpc
        self.id = id
        self.method = method
        self.params = params
        self.result = result
        self.error = error


class MCPError(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(f"MCP Error {code}: {message}")


class MCPErrorCodes:
    INTERNAL_ERROR = -32603
    PARSE_ERROR = -32700


# Transport Layer Abstraction
class MCPTransport(ABC):
    """Abstract base class for MCP transport implementations."""

    @abstractmethod
    async def send(self, message: MCPMessage) -> None:
        """Send a message through the transport."""
        pass

    @abstractmethod
    async def receive(self) -> AsyncGenerator[MCPMessage, None]:
        """Receive messages from the transport."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the transport connection."""
        pass


class MCPStdioTransport(MCPTransport):
    """Standard input/output transport for MCP with cross-platform support."""

    def __init__(self):
        self._closed = False
        self._initialized = False

        # Platform-specific attributes
        if platform.system() == "Windows":
            self.input_queue = None
            self.output_queue = None
            self.input_thread = None
            self.output_thread = None
            self.executor = None
        else:
            self.reader: Optional[asyncio.StreamReader] = None
            self.writer: Optional[asyncio.StreamWriter] = None

    async def initialize(self):
        """Initialize stdio transport with platform-specific handling."""
        if self._initialized:
            return

        if platform.system() == "Windows":
            await self._init_windows()
        else:
            await self._init_unix()

        self._initialized = True

    async def _init_windows(self):
        """Windows-specific initialization using threading."""
        import concurrent.futures

        self.input_queue = asyncio.Queue()
        self.output_queue = queue.Queue()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        # Start background threads for I/O
        self.input_thread = threading.Thread(target=self._input_worker, daemon=True)
        self.output_thread = threading.Thread(target=self._output_worker, daemon=True)

        self.input_thread.start()
        self.output_thread.start()

    async def _init_unix(self):
        """Unix-specific initialization using standard asyncio."""
        self.reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(self.reader)
        loop = asyncio.get_event_loop()
        transport, _ = await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        self.writer = asyncio.StreamWriter(sys.stdout, None, self.reader, loop)

    def _input_worker(self):
        """Worker thread to read from stdin (Windows only)."""
        try:
            loop = asyncio.new_event_loop()
            while not self._closed:
                try:
                    line = sys.stdin.readline()
                    if not line:
                        break

                    # Get the main event loop and schedule the coroutine
                    main_loop = asyncio.get_event_loop()
                    asyncio.run_coroutine_threadsafe(
                        self.input_queue.put(line.strip()),
                        main_loop
                    )
                except Exception as e:
                    # Handle errors gracefully
                    if not self._closed:
                        print(f"Input worker error: {e}", file=sys.stderr)
                    break
        except Exception as e:
            if not self._closed:
                print(f"Input worker initialization error: {e}", file=sys.stderr)

    def _output_worker(self):
        """Worker thread to write to stdout (Windows only)."""
        try:
            while not self._closed:
                try:
                    message = self.output_queue.get(timeout=0.1)
                    if message is None:  # Sentinel to stop
                        break
                    sys.stdout.write(message)
                    sys.stdout.flush()
                    self.output_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    if not self._closed:
                        print(f"Output worker error: {e}", file=sys.stderr)
                    break
        except Exception as e:
            if not self._closed:
                print(f"Output worker initialization error: {e}", file=sys.stderr)

    async def send(self, message: MCPMessage) -> None:
        """Send message to stdout."""
        if self._closed:
            raise MCPError(MCPErrorCodes.INTERNAL_ERROR, "Transport is closed")

        if not self._initialized:
            await self.initialize()

        message_dict = {
            "jsonrpc": message.jsonrpc,
        }

        if message.id is not None:
            message_dict["id"] = message.id
        if message.method is not None:
            message_dict["method"] = message.method
        if message.params is not None:
            message_dict["params"] = message.params
        if message.result is not None:
            message_dict["result"] = message.result
        if message.error is not None:
            message_dict["error"] = message.error

        json_str = json.dumps(message_dict)
        message_with_newline = f"{json_str}\n"

        if platform.system() == "Windows":
            # Use threading approach for Windows
            self.output_queue.put(message_with_newline)
        else:
            # Use standard asyncio for Unix
            if not self.writer:
                raise MCPError(MCPErrorCodes.INTERNAL_ERROR, "Writer not initialized")
            self.writer.write(message_with_newline.encode())
            await self.writer.drain()

    async def receive(self) -> AsyncGenerator[MCPMessage, None]:
        """Receive messages from stdin."""
        if not self._initialized:
            await self.initialize()

        while not self._closed:
            try:
                if platform.system() == "Windows":
                    # Use threading approach for Windows
                    line = await self.input_queue.get()
                    if not line:
                        break
                else:
                    # Use standard asyncio for Unix
                    if not self.reader:
                        raise MCPError(MCPErrorCodes.INTERNAL_ERROR, "Reader not initialized")
                    line_bytes = await self.reader.readline()
                    if not line_bytes:
                        break
                    line = line_bytes.decode().strip()

                if not line:
                    continue

                data = json.loads(line)
                message = MCPMessage(
                    jsonrpc=data.get("jsonrpc", "2.0"),
                    id=data.get("id"),
                    method=data.get("method"),
                    params=data.get("params"),
                    result=data.get("result"),
                    error=data.get("error")
                )
                yield message

            except json.JSONDecodeError as e:
                error_message = MCPMessage(
                    error={
                        "code": MCPErrorCodes.PARSE_ERROR,
                        "message": f"JSON parse error: {str(e)}"
                    }
                )
                yield error_message
            except Exception as e:
                if not self._closed:
                    error_message = MCPMessage(
                        error={
                            "code": MCPErrorCodes.INTERNAL_ERROR,
                            "message": f"Transport error: {str(e)}"
                        }
                    )
                    yield error_message

    async def close(self) -> None:
        """Close the transport."""
        self._closed = True

        if platform.system() == "Windows":
            # Clean up Windows threading resources
            if self.output_queue:
                self.output_queue.put(None)  # Sentinel to stop output worker

            # Wait for threads to finish (with timeout)
            if self.input_thread and self.input_thread.is_alive():
                self.input_thread.join(timeout=1.0)
            if self.output_thread and self.output_thread.is_alive():
                self.output_thread.join(timeout=1.0)

            if self.executor:
                self.executor.shutdown(wait=False)
        else:
            # Clean up Unix asyncio resources
            if self.writer:
                self.writer.close()
                await self.writer.wait_closed()


# Example usage
async def main():
    """Example of how to use the transport."""
    transport = MCPStdioTransport()

    try:
        # Example: send a message
        test_message = MCPMessage(
            method="test",
            params={"data": "hello"},
            id=1
        )
        await transport.send(test_message)

        # Example: receive messages
        async for message in transport.receive():
            print(f"Received: {message.method}")
            # Process message here
            break  # Just for example

    finally:
        await transport.close()


if __name__ == "__main__":
    # Set the selector event loop policy for Windows to avoid the proactor bug
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())