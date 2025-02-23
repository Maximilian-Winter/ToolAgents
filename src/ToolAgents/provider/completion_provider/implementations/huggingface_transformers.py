from threading import Thread
from typing import Any, Union, Generator
import abc
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch

from ToolAgents.provider.completion_provider.completion_interfaces import CompletionEndpoint, AsyncCompletionEndpoint
from ToolAgents.provider.llm_provider import SamplerSetting, ProviderSettings


class TransformersProviderSettings(ProviderSettings):
    def __init__(self):
        # Define sampler settings with their default and neutral values
        samplers = [
            SamplerSetting.create_sampler_setting("temperature", 1.0, 1.0),
            SamplerSetting.create_sampler_setting("top_k", 50, 0),
            SamplerSetting.create_sampler_setting("top_p", 1.0, 1.0),
            SamplerSetting.create_sampler_setting("repetition_penalty", 1.0, 1.0),
        ]

        # Initialize base class with empty tool choice and samplers
        super().__init__(initial_tool_choice="", samplers=samplers)

        # Initialize other default settings
        self.set_extra_request_kwargs(
            do_sample=True,
            pad_token_id=None,
            bos_token_id=None,
            eos_token_id=None,
            use_cache=True,
            num_beams=1,
            early_stopping=False
        )

    def to_dict(self, include: list[str] = None, filter_out: list[str] = None) -> dict[str, Any]:
        """Override to handle the specific requirements of Transformers"""
        result = super().to_dict(include, filter_out)

        # Rename max_tokens to max_new_tokens if present
        if 'max_tokens' in result:
            result['max_new_tokens'] = result.pop('max_tokens')

        return result


class TransformersCompletionEndpoint(CompletionEndpoint):


    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def create_completion(
            self,
            prompt: str,
            settings: TransformersProviderSettings
    ) -> str:
        """Create a completion using the Transformers model"""
        # Prepare generation config
        generation_config = self._prepare_generation_config(settings)

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return self._generate_completion(inputs, generation_config)

    def create_streaming_completion(self, prompt, settings: TransformersProviderSettings) -> Generator[str, None, None]:
        # Prepare generation config
        generation_config = self._prepare_generation_config(settings)

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self._generate_stream(inputs, generation_config)

    def _prepare_generation_config(self, settings: TransformersProviderSettings) -> dict:
        """Prepare generation configuration from settings"""
        config = settings.to_dict(filter_out=['stop_sequences', 'tool_choice', 'stream'])

        # Remove any None values
        return {k: v for k, v in config.items() if v is not None}

    def _generate_completion(self, inputs: dict, generation_config: dict) -> str:
        """Generate a complete response"""
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)

        # Decode and remove prompt
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_length = len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))

        return decoded[prompt_length:].strip()

    def _generate_stream(self, inputs: dict, generation_config: dict) -> Generator[str, None, None]:
        """Stream the generated response token by token"""
        generation_config['streaming'] = True

        with torch.no_grad():
            streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
            generation_kwargs = dict(**inputs, **generation_config, streamer=streamer)

            # Start generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # Yield from streamer
            for text in streamer:
                yield text

    def get_default_settings(self) -> TransformersProviderSettings:
        """Return default settings for the Transformers endpoint"""
        return TransformersProviderSettings()


import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread


class AsyncTransformersCompletionEndpoint(AsyncCompletionEndpoint):
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._thread_pool = ThreadPoolExecutor(max_workers=1)  # For running sync code
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    async def create_completion(
            self,
            prompt: str,
            settings: TransformersProviderSettings
    ) -> str:
        """Create a completion using the Transformers model asynchronously"""
        # Prepare generation config
        generation_config = self._prepare_generation_config(settings)

        # Run tokenization and generation in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        def sync_generate():
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)

            # Decode and remove prompt
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            prompt_length = len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
            return decoded[prompt_length:].strip()

        # Run the synchronous code in a thread pool
        return await loop.run_in_executor(self._thread_pool, sync_generate)

    async def create_streaming_completion(
            self,
            prompt: str,
            settings: TransformersProviderSettings
    ) -> AsyncGenerator[str, None]:
        """Stream the generated response token by token asynchronously"""
        generation_config = self._prepare_generation_config(settings)
        generation_config['streaming'] = True

        # Create queue for async communication
        queue = asyncio.Queue()

        def sync_generate():
            try:
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
                generation_kwargs = dict(**inputs, **generation_config, streamer=streamer)

                # Start generation in a separate thread
                thread = Thread(target=self._generate_and_queue,
                                args=(self.model, generation_kwargs, streamer, queue))
                thread.start()

            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    queue.put(('error', str(e))),
                    asyncio.get_event_loop()
                )

        # Start generation in thread pool
        loop = asyncio.get_event_loop()
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

    def _generate_and_queue(
            self,
            model: AutoModelForCausalLM,
            generation_kwargs: Dict[str, Any],
            streamer: TextIteratorStreamer,
            queue: asyncio.Queue
    ):
        """Helper method to generate tokens and put them in the queue"""
        try:
            # Start the generation
            with torch.no_grad():
                model.generate(**generation_kwargs)

            # Queue up tokens as they're generated
            for text in streamer:
                asyncio.run_coroutine_threadsafe(
                    queue.put(('token', text)),
                    asyncio.get_event_loop()
                )

            # Signal completion
            asyncio.run_coroutine_threadsafe(
                queue.put(('done', None)),
                asyncio.get_event_loop()
            )

        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                queue.put(('error', str(e))),
                asyncio.get_event_loop()
            )

    def _prepare_generation_config(self, settings: TransformersProviderSettings) -> Dict[str, Any]:
        """Prepare generation configuration from settings"""
        config = settings.to_dict(filter_out=['stop_sequences', 'tool_choice', 'stream'])
        # Remove any None values
        return {k: v for k, v in config.items() if v is not None}

    def get_default_settings(self) -> TransformersProviderSettings:
        """Return default settings for the Transformers endpoint"""
        return TransformersProviderSettings()