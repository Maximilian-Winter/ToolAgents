from typing import Generator, Dict, Any, Optional
from llama_cpp import Llama

from ToolAgents.provider import ProviderSettings
from ToolAgents.provider.completion_provider.completion_interfaces import CompletionEndpoint, AsyncCompletionEndpoint
from ToolAgents.provider.llm_provider import SamplerSetting

class LlamaCppPythonSettings(ProviderSettings):
    def __init__(self):
        # Define sampler settings with their default and neutral values
        samplers = [
            SamplerSetting.create_sampler_setting("temperature", 0.8, 1.0),
            SamplerSetting.create_sampler_setting("top_k", 40, 0),
            SamplerSetting.create_sampler_setting("top_p", 0.95, 1.0),
            SamplerSetting.create_sampler_setting("min_p", 0.05, 0.0),
            SamplerSetting.create_sampler_setting("repeat_penalty", 1.1, 1.0),
            SamplerSetting.create_sampler_setting("presence_penalty", 0.0, 0.0),
            SamplerSetting.create_sampler_setting("frequency_penalty", 0.0, 0.0),
        ]

        # Initialize base class with empty tool choice and samplers
        super().__init__(initial_tool_choice="", samplers=samplers)

        # Initialize other default settings
        self.set_extra_request_kwargs(
            echo=False,  # Whether to echo the prompt
            stop=None,  # Stop sequences
            stream=False,  # Whether to stream the response
            grammar=None,  # Grammar to use for completion
            mirostat_mode=0,  # Mirostat sampling mode (0 = disabled, 1 = mirostat, 2 = mirostat 2.0)
            mirostat_tau=5.0,  # Mirostat target entropy
            mirostat_eta=0.1,  # Mirostat learning rate
        )

    def to_dict(self, include: Optional[list[str]] = None, filter_out: Optional[list[str]] = None) -> Dict[str, Any]:
        """Convert settings to llama.cpp format"""
        result = super().to_dict(include, filter_out)

        # Map our generic sampler names to llama.cpp specific names
        if 'repeat_penalty' in result:
            result['repeat_penalty'] = result.pop('repeat_penalty')

        # Handle stop sequences
        if 'stop_sequences' in result:
            result['stop'] = result.pop('stop_sequences')

        return result


class LlamaCppPythonEndpoint(CompletionEndpoint):
    def __init__(self,
                 model_path: str,
                 n_ctx: int = 2048,
                 n_batch: int = 512,
                 n_threads: Optional[int] = None,
                 n_gpu_layers: int = 0):
        """
        Initialize the llama.cpp Python endpoint

        Args:
            model_path: Path to the model file
            n_ctx: Context window size
            n_batch: Batch size for prompt processing
            n_threads: Number of CPU threads to use (None = auto)
            n_gpu_layers: Number of layers to offload to GPU
        """
        super().__init__()
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self._llm = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the llama.cpp model"""
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )

    def create_completion(self, prompt: str, settings: LlamaCppPythonSettings) -> str:
        """Create a completion using llama.cpp Python bindings"""
        completion_kwargs = self._prepare_completion_kwargs(settings, prompt)
        completion_kwargs['stream'] = False

        try:
            output = self._llm(**completion_kwargs)
            return output['choices'][0]['text']
        except Exception as e:
            # Add context to the error
            raise RuntimeError(f"Error during completion generation: {str(e)}") from e

    def create_streaming_completion(self, prompt: str, settings: LlamaCppPythonSettings) -> Generator[str, None, None]:
        """Stream completions using llama.cpp Python bindings"""
        completion_kwargs = self._prepare_completion_kwargs(settings, prompt)
        completion_kwargs['stream'] = True

        try:
            for chunk in self._llm(**completion_kwargs):
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    text = chunk['choices'][0]['text']
                    if text:
                        yield text
        except Exception as e:
            # Add context to the error
            raise RuntimeError(f"Error during stream generation: {str(e)}") from e

    def _prepare_completion_kwargs(self, settings: LlamaCppPythonSettings, prompt: str) -> Dict[str, Any]:
        """Prepare kwargs for llama.cpp completion"""
        kwargs = settings.to_dict(filter_out=["tool_choice"])

        # Add prompt
        kwargs['prompt'] = prompt

        # Remove any None values
        return {k: v for k, v in kwargs.items() if v is not None}

    def get_default_settings(self) -> LlamaCppPythonSettings:
        """Return default settings for llama.cpp"""
        return LlamaCppPythonSettings()

    def __del__(self):
        """Cleanup when the endpoint is destroyed"""
        if self._llm is not None:
            del self._llm


from typing import AsyncGenerator, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
from llama_cpp import Llama




class AsyncLlamaCppPythonEndpoint(AsyncCompletionEndpoint):
    def __init__(self,
                 model_path: str,
                 n_ctx: int = 2048,
                 n_batch: int = 512,
                 n_threads: Optional[int] = None,
                 n_gpu_layers: int = 0):
        """
        Initialize the async llama.cpp Python endpoint

        Args:
            model_path: Path to the model file
            n_ctx: Context window size
            n_batch: Batch size for prompt processing
            n_threads: Number of CPU threads to use (None = auto)
            n_gpu_layers: Number of layers to offload to GPU
        """
        super().__init__()
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self._thread_pool = ThreadPoolExecutor(max_workers=1)  # For running sync llama.cpp code
        self._llm = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the llama.cpp model"""
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers
        )

    async def create_completion(self, prompt: str, settings: ProviderSettings) -> str:
        """Create a completion using llama.cpp Python bindings asynchronously"""
        loop = asyncio.get_event_loop()

        def sync_generate():
            completion_kwargs = self._prepare_completion_kwargs(settings, prompt)
            completion_kwargs['stream'] = False
            output = self._llm(**completion_kwargs)
            return output['choices'][0]['text']

        # Run the synchronous code in a thread pool
        return await loop.run_in_executor(self._thread_pool, sync_generate)

    async def create_streaming_completion(self, prompt: str, settings: ProviderSettings) -> AsyncGenerator[str, None]:
        """Stream completions using llama.cpp Python bindings asynchronously"""
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue()

        def sync_generate():
            try:
                completion_kwargs = self._prepare_completion_kwargs(settings, prompt)
                completion_kwargs['stream'] = True

                for chunk in self._llm(**completion_kwargs):
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        text = chunk['choices'][0]['text']
                        if text:
                            asyncio.run_coroutine_threadsafe(
                                queue.put(('token', text)),
                                loop
                            )

                # Signal completion
                asyncio.run_coroutine_threadsafe(
                    queue.put(('done', None)),
                    loop
                )

            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    queue.put(('error', str(e))),
                    loop
                )

        # Start generation in thread pool
        await loop.run_in_executor(self._thread_pool, sync_generate)

        # Yield tokens as they become available
        try:
            while True:
                msg_type, content = await queue.get()
                if msg_type == 'error':
                    raise RuntimeError(content)
                elif msg_type == 'token':
                    yield content
                elif msg_type == 'done':
                    break
                queue.task_done()
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            pass

    def _prepare_completion_kwargs(self, settings: ProviderSettings, prompt: str) -> Dict[str, Any]:
        """Prepare kwargs for llama.cpp completion"""
        kwargs = settings.to_dict(filter_out=["tool_choice"])

        # Add prompt
        kwargs['prompt'] = prompt

        # Remove any None values
        return {k: v for k, v in kwargs.items() if v is not None}

    def get_default_settings(self) -> LlamaCppPythonSettings:
        """Return default settings for llama.cpp"""
        return LlamaCppPythonSettings()