#!/usr/bin/env python3
"""
Big Three Realtime Agents - WSL Tool Service
============================================

This module hosts the Claude and Gemini agent execution environment inside
WSL. It exposes a lightweight bridge that can be invoked from the Windows
voice controller (``big_3_Windows.py``) to execute tooling operations such as
creating agents, running browser automation, or reading files.
"""

# /// script
# dependencies = [
#     "websocket-client",
#     "pyaudio",
#     "python-dotenv",
#     "rich",
#     "claude-agent-sdk",
#     "google-genai",
#     "playwright",
#     "numpy",
#     "pynput",
# ]
# ///

import os
import json
import logging
import threading
import argparse
import asyncio
import textwrap
import urllib.request
import urllib.error
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

# Environment setup
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    raise ImportError(
        "dotenv not found. Please install it with `pip install python-dotenv`"
    )

# Claude Agent SDK imports
try:
    from claude_agent_sdk import (
        query,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        HookMatcher,
        HookContext,
        AssistantMessage,
        SystemMessage,
        UserMessage,
        ResultMessage,
        TextBlock,
        ThinkingBlock,
        ToolUseBlock,
        ToolResultBlock,
        tool,
        create_sdk_mcp_server,
    )
except ImportError as exc:
    raise ImportError(
        "claude-agent-sdk not found. Install with `pip install claude-agent-sdk`."
    ) from exc

# Gemini imports
try:
    from google import genai
    from google.genai import types
    from google.genai.types import Content, Part
except ImportError as exc:
    raise ImportError(
        "google-genai not found. Install with `pip install google-genai`."
    ) from exc

# Playwright imports
try:
    from playwright.sync_api import sync_playwright, Page
except ImportError as exc:
    raise ImportError(
        "playwright not found. Install with `pip install playwright` and run `playwright install`."
    ) from exc

# ================================================================
# Constants
# ================================================================

# OpenAI Realtime API
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
REALTIME_MODEL_DEFAULT = os.environ.get("REALTIME_MODEL", "gpt-realtime-2025-08-28")
REALTIME_MODEL_MINI = "gpt-realtime-mini-2025-10-06"
REALTIME_API_URL_TEMPLATE = "wss://api.openai.com/v1/realtime?model={model}"
REALTIME_VOICE_CHOICE = os.environ.get("REALTIME_AGENT_VOICE", "shimmer")
BROWSER_TOOL_STARTING_URL = os.environ.get(
    "BROWSER_TOOL_STARTING_URL", "localhost:3333"
)

# Audio configuration
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000

# Claude Code configuration
DEFAULT_CLAUDE_MODEL = os.environ.get(
    "CLAUDE_AGENT_MODEL", "claude-sonnet-4-5-20250929"
)
ENGINEER_NAME = os.environ.get("ENGINEER_NAME", "Dan")
REALTIME_ORCH_AGENT_NAME = os.environ.get("REALTIME_ORCH_AGENT_NAME", "ada")
CLAUDE_CODE_TOOL = "claude_code"
CLAUDE_CODE_TOOL_SLUG = "claude_code"
AGENTIC_CODING_TYPE = "agentic_coding"

# Gemini Computer Use configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-computer-use-preview-10-2025"
GEMINI_TOOL = "gemini"
GEMINI_TOOL_SLUG = "gemini"
AGENTIC_BROWSERING_TYPE = "agentic_browsering"
SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900

# Common agent configuration (defined after AGENT_WORKING_DIRECTORY below)
PROMPTS_DIR = Path(__file__).parent / "prompts"

# Agent working directory - set to content-gen app
AGENT_WORKING_DIRECTORY = Path(__file__).parent.parent / "content-gen"

# Set AGENTS_BASE_DIR relative to working directory for consolidated outputs
AGENTS_BASE_DIR = AGENT_WORKING_DIRECTORY / "agents"
CLAUDE_CODE_REGISTRY_PATH = AGENTS_BASE_DIR / CLAUDE_CODE_TOOL_SLUG / "registry.json"
GEMINI_REGISTRY_PATH = AGENTS_BASE_DIR / GEMINI_TOOL_SLUG / "registry.json"

# Console for rich output
console = Console()


# ================================================================
# GeminiBrowserAgent - Browser automation with Gemini Computer Use
# ================================================================


class GeminiBrowserAgent:
    """
    Browser automation agent powered by Gemini Computer Use API.

    Handles web browsing, navigation, and interaction tasks using
    Gemini's vision and action planning capabilities with Playwright.
    """

    def __init__(self, logger=None):
        """Initialize browser agent."""
        self.logger = logger or logging.getLogger("GeminiBrowserAgent")

        # Validate Gemini API key
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)

        # Browser automation state
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

        # Screenshot session setup - persistent for entire browser session
        self.session_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        )
        self.screenshot_dir = Path("output_screenshots") / self.session_id
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_counter = 0

        # Registry management for browser agents
        self.registry_lock = threading.Lock()
        self.agent_registry = self._load_agent_registry()

        self.logger.info(f"Browser session ID: {self.session_id}")
        self.logger.info(f"Screenshots will be saved to: {self.screenshot_dir}")
        self.logger.info("Initialized GeminiBrowserAgent")

    # ------------------------------------------------------------------ #
    # Agent registry helpers
    # ------------------------------------------------------------------ #

    def _load_agent_registry(self) -> Dict[str, Any]:
        """Load agent registry from disk."""
        if not GEMINI_REGISTRY_PATH.exists():
            return {"agents": {}}

        try:
            with GEMINI_REGISTRY_PATH.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if "agents" not in data:
                    data["agents"] = {}
                return data
        except Exception as exc:
            self.logger.error(f"Failed to load agent registry: {exc}")
            return {"agents": {}}

    def _save_agent_registry(self):
        """Save agent registry to disk."""
        GEMINI_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with GEMINI_REGISTRY_PATH.open("w", encoding="utf-8") as fh:
                json.dump(self.agent_registry, fh, indent=2)
        except Exception as exc:
            self.logger.error(f"Failed to save agent registry: {exc}")

    def _register_agent(self, agent_name: str, metadata: Dict[str, Any]):
        """Register a browser agent in the registry."""
        with self.registry_lock:
            self.agent_registry.setdefault("agents", {})[agent_name] = {
                "tool": GEMINI_TOOL,
                "type": AGENTIC_BROWSERING_TYPE,
                "created_at": metadata.get(
                    "created_at", datetime.now(timezone.utc).isoformat()
                ),
                "session_id": metadata.get("session_id", self.session_id),
            }
            self._save_agent_registry()

    def _get_agent_by_name(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get agent metadata by name."""
        return self.agent_registry.get("agents", {}).get(agent_name)

    def _agent_directory(self, agent_name: str) -> Path:
        """Get agent working directory path."""
        return AGENTS_BASE_DIR / GEMINI_TOOL_SLUG / agent_name

    # ------------------------------------------------------------------ #
    # Browser automation
    # ------------------------------------------------------------------ #

    def setup_browser(self):
        """Initialize Playwright browser."""
        try:
            self.logger.info("Initializing browser...")
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=False)
            self.context = self.browser.new_context(
                viewport={"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT}
            )
            self.page = self.context.new_page()
            self.logger.info("Browser ready!")
        except Exception as e:
            self.logger.error(f"Failed to initialize browser: {e}")
            raise

    def cleanup_browser(self):
        """Clean up Playwright browser resources."""
        try:
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            self.logger.info("Browser cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Browser cleanup error: {e}")

    def execute_task(
        self, task: str, url: Optional[str] = BROWSER_TOOL_STARTING_URL
    ) -> Dict[str, Any]:
        """
        Execute a browser automation task.

        Args:
            task: Description of the browsing task to perform
            url: Optional starting URL

        Returns:
            Dictionary with ok status and either data or error
        """
        try:
            self.logger.info(f"Task: {task}")
            self.logger.info(f"Starting URL: {url or BROWSER_TOOL_STARTING_URL}")
            self.logger.info(f"Session ID: {self.session_id}")

            # Setup browser if not already done
            if not self.page:
                self.setup_browser()

            # Navigate to starting URL if provided
            if url:
                self.page.goto(url, wait_until="networkidle", timeout=10000)
                self.logger.info(f"Navigated to: {url}")
            else:
                # Start with a search engine
                self.page.goto(
                    "https://www.google.com", wait_until="networkidle", timeout=10000
                )
                self.logger.info("Starting from Google")

            # Run the browser automation loop
            result = self._run_browser_automation_loop(task)

            self.logger.info(
                f"Task completed! Screenshots saved to: {self.screenshot_dir}"
            )

            return {
                "ok": True,
                "data": result,
                "screenshot_dir": str(self.screenshot_dir),
            }

        except Exception as exc:
            self.logger.exception("Browser automation failed")
            return {"ok": False, "error": str(exc)}

    def _run_browser_automation_loop(self, task: str, max_turns: int = 30) -> str:
        """
        Run the Gemini Computer Use agent loop to complete the task.

        Args:
            task: The browsing task to complete
            max_turns: Maximum number of agent turns

        Returns:
            The final result as a string
        """
        # Configure Gemini with Computer Use
        config = types.GenerateContentConfig(
            tools=[
                types.Tool(
                    computer_use=types.ComputerUse(
                        environment=types.Environment.ENVIRONMENT_BROWSER
                    )
                )
            ],
        )

        # Initial screenshot
        initial_screenshot = self.page.screenshot(type="png")

        # Save initial screenshot
        timestamp = datetime.now().strftime("%H%M%S")
        screenshot_path = (
            self.screenshot_dir
            / f"step_{self.screenshot_counter:02d}_initial_{timestamp}.png"
        )
        self.page.screenshot(path=str(screenshot_path))
        self.logger.info(f"Saved initial screenshot: {screenshot_path}")
        self.screenshot_counter += 1

        # Build initial contents
        contents = [
            Content(
                role="user",
                parts=[
                    Part(text=task),
                    Part.from_bytes(data=initial_screenshot, mime_type="image/png"),
                ],
            )
        ]

        self.logger.info(f"Starting browser automation loop for task: {task}")

        # Agent loop
        for turn in range(max_turns):
            self.logger.info(f"Turn {turn + 1}/{max_turns}")

            try:
                # Get response from Gemini
                response = self.gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=config,
                )

                candidate = response.candidates[0]
                contents.append(candidate.content)

                # Check if there are function calls
                has_function_calls = any(
                    part.function_call for part in candidate.content.parts
                )

                if not has_function_calls:
                    # No more actions - extract final text response
                    text_response = " ".join(
                        [part.text for part in candidate.content.parts if part.text]
                    )
                    self.logger.info(f"Agent finished: {text_response}")

                    console.print(Panel(text_response, title="GeminiBrowserAgent"))

                    # Save final screenshot
                    timestamp = datetime.now().strftime("%H%M%S")
                    screenshot_path = (
                        self.screenshot_dir
                        / f"step_{self.screenshot_counter:02d}_final_{timestamp}.png"
                    )
                    self.page.screenshot(path=str(screenshot_path))
                    self.logger.info(f"Saved final screenshot: {screenshot_path}")
                    self.screenshot_counter += 1

                    return text_response

                # Execute function calls
                self.logger.info("Executing browser actions...")
                results = self._execute_gemini_function_calls(candidate)

                # Get function responses with new screenshot
                function_responses = self._get_gemini_function_responses(results)

                # Save screenshot after actions
                timestamp = datetime.now().strftime("%H%M%S")
                screenshot_path = (
                    self.screenshot_dir
                    / f"step_{self.screenshot_counter:02d}_{timestamp}.png"
                )
                self.page.screenshot(path=str(screenshot_path))
                self.logger.info(f"Saved screenshot: {screenshot_path}")
                self.screenshot_counter += 1

                # Add function responses to contents
                contents.append(
                    Content(
                        role="user",
                        parts=[Part(function_response=fr) for fr in function_responses],
                    )
                )

            except Exception as e:
                self.logger.error(f"Error in browser automation loop: {e}")
                raise

        # If we hit max turns, return what we have
        return f"Task reached maximum turns ({max_turns}). Please check browser state."

    def _execute_gemini_function_calls(self, candidate) -> list:
        """Execute Gemini Computer Use function calls using Playwright."""
        results = []
        function_calls = [
            part.function_call for part in candidate.content.parts if part.function_call
        ]

        for function_call in function_calls:
            fname = function_call.name
            args = function_call.args
            self.logger.info(f"Executing Gemini action: {fname}")

            action_result = {}

            try:
                if fname == "open_web_browser":
                    pass  # Already open
                elif fname == "wait_5_seconds":
                    time.sleep(5)
                elif fname == "go_back":
                    self.page.go_back()
                elif fname == "go_forward":
                    self.page.go_forward()
                elif fname == "search":
                    self.page.goto("https://www.google.com")
                elif fname == "navigate":
                    self.page.goto(args["url"], wait_until="networkidle", timeout=10000)
                elif fname == "click_at":
                    actual_x = self._denormalize_x(args["x"])
                    actual_y = self._denormalize_y(args["y"])
                    self.page.mouse.click(actual_x, actual_y)
                elif fname == "hover_at":
                    actual_x = self._denormalize_x(args["x"])
                    actual_y = self._denormalize_y(args["y"])
                    self.page.mouse.move(actual_x, actual_y)
                elif fname == "type_text_at":
                    actual_x = self._denormalize_x(args["x"])
                    actual_y = self._denormalize_y(args["y"])
                    text = args["text"]
                    press_enter = args.get("press_enter", True)
                    clear_before = args.get("clear_before_typing", True)

                    self.page.mouse.click(actual_x, actual_y)
                    if clear_before:
                        self.page.keyboard.press("Meta+A")
                        self.page.keyboard.press("Backspace")
                    self.page.keyboard.type(text)
                    if press_enter:
                        self.page.keyboard.press("Enter")
                elif fname == "key_combination":
                    keys = args["keys"]
                    self.page.keyboard.press(keys)
                elif fname == "scroll_document":
                    direction = args["direction"]
                    if direction == "down":
                        self.page.keyboard.press("PageDown")
                    elif direction == "up":
                        self.page.keyboard.press("PageUp")
                    elif direction == "left":
                        self.page.keyboard.press("ArrowLeft")
                    elif direction == "right":
                        self.page.keyboard.press("ArrowRight")
                elif fname == "scroll_at":
                    actual_x = self._denormalize_x(args["x"])
                    actual_y = self._denormalize_y(args["y"])
                    direction = args["direction"]
                    magnitude = args.get("magnitude", 800)

                    # Scroll by moving to position and using wheel
                    self.page.mouse.move(actual_x, actual_y)
                    scroll_amount = int(magnitude * SCREEN_HEIGHT / 1000)
                    if direction == "down":
                        self.page.mouse.wheel(0, scroll_amount)
                    elif direction == "up":
                        self.page.mouse.wheel(0, -scroll_amount)
                    elif direction == "left":
                        self.page.mouse.wheel(-scroll_amount, 0)
                    elif direction == "right":
                        self.page.mouse.wheel(scroll_amount, 0)
                elif fname == "drag_and_drop":
                    x = self._denormalize_x(args["x"])
                    y = self._denormalize_y(args["y"])
                    dest_x = self._denormalize_x(args["destination_x"])
                    dest_y = self._denormalize_y(args["destination_y"])

                    self.page.mouse.move(x, y)
                    self.page.mouse.down()
                    self.page.mouse.move(dest_x, dest_y)
                    self.page.mouse.up()
                else:
                    self.logger.warning(f"Unimplemented action: {fname}")

                # Wait for potential navigations/renders
                self.page.wait_for_load_state(timeout=5000)
                time.sleep(1)

            except Exception as e:
                self.logger.error(f"Error executing {fname}: {e}")
                action_result = {"error": str(e)}

            results.append((fname, action_result))

        return results

    def _get_gemini_function_responses(self, results: list):
        """Generate function responses with current screenshot."""
        screenshot_bytes = self.page.screenshot(type="png")
        current_url = self.page.url
        function_responses = []

        for name, result in results:
            response_data = {"url": current_url}
            response_data.update(result)
            function_responses.append(
                types.FunctionResponse(
                    name=name,
                    response=response_data,
                    parts=[
                        types.FunctionResponsePart(
                            inline_data=types.FunctionResponseBlob(
                                mime_type="image/png", data=screenshot_bytes
                            )
                        )
                    ],
                )
            )

        return function_responses

    def _denormalize_x(self, x: int) -> int:
        """Convert normalized x coordinate (0-999) to actual pixel coordinate."""
        return int(x / 1000 * SCREEN_WIDTH)

    def _denormalize_y(self, y: int) -> int:
        """Convert normalized y coordinate (0-999) to actual pixel coordinate."""
        return int(y / 1000 * SCREEN_HEIGHT)


# ================================================================
# ClaudeCodeAgenticCoder - Claude Code agent orchestration
# ================================================================


class ClaudeCodeAgenticCoder:
    """
    Manages Claude Code agents for software development tasks.

    Handles agent creation, command dispatch, and result retrieval.
    Each agent maintains session continuity for context-aware development.
    """

    def __init__(self, logger=None, browser_agent=None):
        """Initialize agentic coder manager."""
        self.logger = logger or logging.getLogger("ClaudeCodeAgenticCoder")
        self.browser_agent = browser_agent

        self.registry_lock = threading.Lock()
        self.agent_registry = self._load_agent_registry()

        self.background_threads: list[threading.Thread] = []

        self.logger.info("Initialized ClaudeCodeAgenticCoder")

    # ------------------------------------------------------------------ #
    # Agent registry helpers
    # ------------------------------------------------------------------ #

    def _load_agent_registry(self) -> Dict[str, Any]:
        """Load agent registry from disk."""
        if not CLAUDE_CODE_REGISTRY_PATH.exists():
            return {"agents": {}}

        try:
            with CLAUDE_CODE_REGISTRY_PATH.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if "agents" not in data:
                    data["agents"] = {}
                return data
        except Exception as exc:
            self.logger.error(f"Failed to load agent registry: {exc}")
            return {"agents": {}}

    def _save_agent_registry(self):
        """Save agent registry to disk."""
        CLAUDE_CODE_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with CLAUDE_CODE_REGISTRY_PATH.open("w", encoding="utf-8") as fh:
                json.dump(self.agent_registry, fh, indent=2)
        except Exception as exc:
            self.logger.error(f"Failed to save agent registry: {exc}")

    def _register_agent(
        self, agent_name: str, session_id: str, metadata: Dict[str, Any]
    ):
        """Register an agent in the registry."""
        with self.registry_lock:
            self.agent_registry.setdefault("agents", {})[agent_name] = {
                "session_id": session_id,
                "tool": metadata.get("tool", CLAUDE_CODE_TOOL),
                "type": metadata.get("type", AGENTIC_CODING_TYPE),
                "created_at": metadata.get(
                    "created_at", datetime.now(timezone.utc).isoformat()
                ),
                "working_dir": metadata.get(
                    "working_dir", str(AGENT_WORKING_DIRECTORY)
                ),
                "operator_files": metadata.get("operator_files", []),
            }
            self._save_agent_registry()

    def _get_agent_by_name(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get agent metadata by name."""
        return self.agent_registry.get("agents", {}).get(agent_name)

    def _agent_directory(self, agent_name: str) -> Path:
        """Get agent working directory path."""
        return AGENTS_BASE_DIR / CLAUDE_CODE_TOOL_SLUG / agent_name

    # ------------------------------------------------------------------ #
    # Browser tool for MCP
    # ------------------------------------------------------------------ #

    def _create_browser_tool(self, agent_name: str):
        """Create browser_use tool for Claude agents with agent-specific screenshot directory."""

        @tool(
            "browser_use",
            "Automate web validation tasks. Use this to validate your frontend work. Can navigate websites, interact with web pages, extract data, confirm (or reject) the work is done correctly, and perform complex multi-step validation tasks.",
            {"task": str, "url": str},
        )
        async def browser_use_tool(args: dict[str, Any]) -> dict[str, Any]:
            """Execute browser automation task with agent-specific screenshot storage."""
            task = args.get("task", "")
            url = args.get("url")

            if not self.browser_agent:
                return {
                    "content": [
                        {"type": "text", "text": "Browser agent not available."}
                    ],
                    "isError": True,
                }

            # Create agent-specific screenshot directory
            session_id = (
                datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
            )
            agent_browser_dir = (
                AGENTS_BASE_DIR
                / CLAUDE_CODE_TOOL_SLUG
                / agent_name
                / "browser_tool"
                / session_id
            )
            agent_browser_dir.mkdir(parents=True, exist_ok=True)

            # Create a temporary browser agent instance with agent-specific screenshot dir
            temp_browser = GeminiBrowserAgent(logger=self.logger)
            temp_browser.screenshot_dir = agent_browser_dir
            temp_browser.session_id = session_id
            temp_browser.screenshot_counter = 0

            # Run browser task in thread pool to avoid Playwright sync API conflict with asyncio
            import concurrent.futures

            loop = asyncio.get_event_loop()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor, lambda: temp_browser.execute_task(task, url)
                )

            # Cleanup browser
            try:
                temp_browser.cleanup_browser()
            except:
                pass

            if result.get("ok"):
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Browser task completed!\n\nResult:\n{result.get('data')}\n\nScreenshots: {result.get('screenshot_dir')}",
                        }
                    ]
                }
            else:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Browser task failed: {result.get('error')}",
                        }
                    ],
                    "isError": True,
                }

        return browser_use_tool

    # ------------------------------------------------------------------ #
    # Prompt helpers
    # ------------------------------------------------------------------ #

    def _read_prompt(self, relative_path: str) -> str:
        """Read prompt file from super_agent directory."""
        prompt_path = PROMPTS_DIR / "super_agent" / relative_path
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        try:
            return prompt_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            raise RuntimeError(f"Failed to read prompt {relative_path}: {exc}") from exc

    def _render_prompt(self, relative_path: str, **kwargs: Any) -> str:
        """Render prompt template with variables."""
        template = self._read_prompt(relative_path)
        if kwargs:
            return template.format(**kwargs)
        return template

    # ------------------------------------------------------------------ #
    # Agent observability
    # ------------------------------------------------------------------ #

    def _send_observability_event(
        self,
        agent_name: str,
        hook_type: str,
        session_id: str,
        payload: dict,
        summary: Optional[str] = None,
    ) -> None:
        """Send observability event to monitoring server (fails silently)."""
        try:
            event_data = {
                "source_app": f"big-three-agents: {agent_name}",
                "session_id": session_id,
                "hook_event_type": hook_type,
                "payload": payload,
                "timestamp": int(datetime.now().timestamp() * 1000),
            }

            # Add summary if available
            if summary:
                event_data["summary"] = summary

            req = urllib.request.Request(
                "http://localhost:4000/events",
                data=json.dumps(event_data).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "BigThreeAgents/1.0",
                },
            )

            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status != 200:
                    self.logger.debug(
                        f"Observability event returned {response.status} for {agent_name}"
                    )

        except urllib.error.URLError as e:
            self.logger.debug(f"Observability event failed for {agent_name}: {e}")
        except Exception as e:
            self.logger.debug(f"Observability event error for {agent_name}: {e}")

    def _create_observability_hook(
        self,
        agent_name: str,
        hook_type: str,
        session_id_holder: dict,
        summarize: bool = True,
    ) -> callable:
        """Create observability hook for any hook type with optional summarization."""

        async def hook(
            input_data: Dict[str, Any],
            tool_use_id: str | None,
            context: HookContext,
        ) -> Dict[str, Any]:
            session_id = session_id_holder.get("session_id", "unknown")

            # Generate summary if enabled
            event_summary = None
            if summarize:
                event_summary = await self._generate_event_summary(
                    agent_name, hook_type, input_data
                )

            # Send event with optional summary
            self._send_observability_event(
                agent_name, hook_type, session_id, input_data, event_summary
            )
            return {}  # Allow all operations

        return hook

    async def _generate_event_summary(
        self, agent_name: str, hook_type: str, input_data: Dict[str, Any]
    ) -> Optional[str]:
        """Generate AI summary of event using Claude Agent SDK."""
        try:
            # Build summary prompt
            tool_name = input_data.get("tool_name", "N/A")
            tool_input = input_data.get("tool_input", {})

            # Extract key context based on tool type
            context_parts = []
            if tool_name == "Bash":
                command = tool_input.get("command", "")[:100]
                context_parts.append(f"Command: {command}")
            elif tool_name in ["Read", "Edit", "Write"]:
                file_path = tool_input.get("file_path", "")
                context_parts.append(f"File: {file_path}")

            context = (
                " | ".join(context_parts) if context_parts else "No specific context"
            )

            # Load prompts from files
            system_prompt = self._read_prompt("event_summarizer_system_prompt.md")
            user_prompt = self._render_prompt(
                "event_summarizer_user_prompt.md",
                AGENT_NAME=agent_name,
                HOOK_TYPE=hook_type,
                TOOL_NAME=tool_name,
                CONTEXT=context,
            )

            # Use Claude Agent SDK query for fast summary
            options = ClaudeAgentOptions(
                model="claude-3-5-haiku-20241022",  # Fast model
                system_prompt=system_prompt,
            )

            chunks = []
            async for message in query(prompt=user_prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            chunks.append(block.text)

            summary = "".join(chunks).strip()
            return summary if summary else None

        except Exception as e:
            self.logger.debug(f"Summary generation failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Public API - Tool implementations
    # ------------------------------------------------------------------ #

    def list_agents(self) -> Dict[str, Any]:
        """List all registered agents."""
        agents_payload: list[Dict[str, Any]] = []
        for name, data in sorted(self.agent_registry.get("agents", {}).items()):
            agents_payload.append(
                {
                    "name": name,
                    "session_id": data.get("session_id"),
                    "tool": data.get("tool"),
                    "type": data.get("type"),
                    "created_at": data.get("created_at"),
                    "working_dir": data.get("working_dir"),
                    "operator_files": data.get("operator_files", []),
                }
            )
        return {"ok": True, "agents": agents_payload}

    def create_agent(
        self,
        tool: str = CLAUDE_CODE_TOOL,
        agent_type: str = AGENTIC_CODING_TYPE,
        agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new Claude Code agent."""
        # Validate tool
        if tool != CLAUDE_CODE_TOOL:
            return {
                "ok": False,
                "error": f"Unsupported tool '{tool}'. Only '{CLAUDE_CODE_TOOL}' is supported by this handler.",
            }

        # Validate agent type
        if agent_type != AGENTIC_CODING_TYPE:
            return {
                "ok": False,
                "error": f"Unsupported agent type '{agent_type}'. Only '{AGENTIC_CODING_TYPE}' is supported by this handler.",
            }

        preferred_name = agent_name.strip() if agent_name else None
        if preferred_name and self._get_agent_by_name(preferred_name):
            return {
                "ok": False,
                "error": (
                    f"Agent '{preferred_name}' already exists. Choose a different name or omit agent_name."
                ),
            }

        try:
            agent_info = asyncio.run(
                self._create_new_agent_async(
                    tool=tool, agent_type=agent_type, agent_name=preferred_name
                )
            )
        except Exception as exc:
            self.logger.exception("create_agent failed")
            return {"ok": False, "error": f"Failed to create agent: {exc}"}

        return {
            "ok": True,
            "agent_name": agent_info["name"],
            "session_id": agent_info["session_id"],
        }

    def command_agent(self, agent_name: str, prompt: str) -> Dict[str, Any]:
        """Dispatch command to a Claude Code agent."""
        agent = self._get_agent_by_name(agent_name)
        if not agent:
            return {
                "ok": False,
                "error": f"Agent '{agent_name}' not found. Create it first.",
            }

        # Validate this is a Claude Code agent
        if agent.get("type") != AGENTIC_CODING_TYPE:
            return {
                "ok": False,
                "error": f"Agent '{agent_name}' is not a {AGENTIC_CODING_TYPE} agent. Wrong handler.",
            }

        # Prepare operator file and dispatch command
        try:
            operator_path = asyncio.run(
                self._prepare_operator_file(name=agent_name, prompt=prompt)
            )
        except Exception as exc:
            self.logger.exception("Failed to prepare operator file")
            return {"ok": False, "error": f"Could not prepare operator log: {exc}"}

        thread = threading.Thread(
            target=self._run_agent_command_thread,
            args=(agent_name, prompt, operator_path),
            daemon=True,
        )
        thread.start()
        self.background_threads.append(thread)

        return {"ok": True, "operator_file": str(operator_path)}

    def check_agent_result(
        self, agent_name: str, operator_file_name: str
    ) -> Dict[str, Any]:
        """Read operator status report."""
        agent_dir = self._agent_directory(agent_name)
        operator_path = agent_dir / operator_file_name

        if not operator_path.exists():
            return {"ok": False, "error": f"Operator file not found: {operator_path}"}

        try:
            content = operator_path.read_text(encoding="utf-8")
        except Exception as exc:
            self.logger.error(f"Failed to read operator file: {exc}")
            return {"ok": False, "error": f"Failed to read operator file: {exc}"}

        return {"ok": True, "content": content}

    def delete_agent(self, agent_name: str) -> Dict[str, Any]:
        """Delete an agent."""
        agent = self._get_agent_by_name(agent_name)
        if not agent:
            return {
                "ok": False,
                "error": f"Agent '{agent_name}' not found. Nothing to delete.",
            }

        warnings: list[str] = []

        with self.registry_lock:
            self.agent_registry.get("agents", {}).pop(agent_name, None)
            try:
                self._save_agent_registry()
            except Exception as exc:
                warnings.append(f"Failed to update registry: {exc}")

        agent_dir = self._agent_directory(agent_name)
        if agent_dir.exists():
            try:
                shutil.rmtree(agent_dir)
            except Exception as exc:
                warnings.append(f"Failed to remove directory {agent_dir}: {exc}")

        payload: Dict[str, Any] = {"ok": True, "agent_name": agent_name}
        if warnings:
            payload["warnings"] = warnings
        return payload

    # ------------------------------------------------------------------ #
    # Async Claude agent operations
    # ------------------------------------------------------------------ #

    async def _create_new_agent_async(
        self, tool: str, agent_type: str, agent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create new Claude Code agent asynchronously."""
        existing_names = list(self.agent_registry.get("agents", {}).keys())
        if agent_name:
            final_name = agent_name
        else:
            candidate_name = await self._generate_agent_name(existing_names)
            final_name = await self._dedupe_agent_name(candidate_name)

        agent_dir = self._agent_directory(final_name)
        agent_dir.mkdir(parents=True, exist_ok=True)

        system_prompt_text = self._render_prompt(
            "agentic_coder_system_prompt_system_prompt.md",
            OPERATOR_FILE="(assigned per task)",
            WORKING_DIR=str(AGENT_WORKING_DIRECTORY),
        )

        # Session ID holder for hooks
        session_id_holder = {"session_id": "unknown"}

        # Create observability hooks
        all_hook_types = [
            "PreToolUse",
            "PostToolUse",
            "Notification",
            "UserPromptSubmit",
            "Stop",
            "SubagentStop",
            "PreCompact",
            "SessionStart",
            "SessionEnd",
        ]

        hooks = {
            hook_type: [
                HookMatcher(
                    hooks=[
                        self._create_observability_hook(
                            final_name, hook_type, session_id_holder
                        )
                    ]
                )
            ]
            for hook_type in all_hook_types
        }

        # Create browser tool if available
        mcp_servers = {}
        allowed_tools_list = [
            "Read",
            "Write",
            "Edit",
            "Bash",
            "Glob",
            "Grep",
            "Task",
            "WebFetch",
            "WebSearch",
            "BashOutput",
            "SlashCommand",
            "TodoWrite",
        ]
        disallowed_tools_list = ["KillShell", "NotebookEdit", "ExitPlanMode"]

        if self.browser_agent:
            browser_tool = self._create_browser_tool(final_name)
            browser_server = create_sdk_mcp_server(
                name="browser", version="1.0.0", tools=[browser_tool]
            )
            mcp_servers["browser"] = browser_server
            allowed_tools_list.append("mcp__browser__browser_use")

        options = ClaudeAgentOptions(
            system_prompt={
                "type": "preset",
                "preset": "claude_code",
                "append": system_prompt_text,
            },
            model=DEFAULT_CLAUDE_MODEL,
            cwd=str(AGENT_WORKING_DIRECTORY),
            permission_mode="bypassPermissions",
            setting_sources=["project"],
            hooks=hooks,
            mcp_servers=mcp_servers,
            allowed_tools=allowed_tools_list,
            disallowed_tools=disallowed_tools_list,
        )

        # Simple greeting - system prompt already has instructions
        greeting = f"Hi, you are {final_name}, a {agent_type} agent. Please acknowledge you're ready and briefly introduce yourself."

        session_id: Optional[str] = None
        transcript: list[str] = []

        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(greeting)

                async for message in client.receive_response():
                    # Log all message types
                    if isinstance(message, UserMessage):
                        self.logger.info(
                            f"[{final_name}] UserMessage: {message.content}"
                        )
                    elif isinstance(message, SystemMessage):
                        self.logger.info(
                            f"[{final_name}] SystemMessage: subtype={message.subtype}, data={message.data}"
                        )
                    elif isinstance(message, AssistantMessage):
                        self.logger.info(
                            f"[{final_name}] AssistantMessage: model={message.model}, blocks={len(message.content)}"
                        )
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                transcript.append(block.text)
                                self.logger.info(
                                    f"[{final_name}] TextBlock: {block.text}"
                                )
                            elif isinstance(block, ThinkingBlock):
                                self.logger.info(
                                    f"[{final_name}] ThinkingBlock: {block.thinking}"
                                )
                            elif isinstance(block, ToolUseBlock):
                                self.logger.info(
                                    f"[{final_name}] ToolUseBlock: name={block.name}, id={block.id}, input={block.input}"
                                )
                            elif isinstance(block, ToolResultBlock):
                                self.logger.info(
                                    f"[{final_name}] ToolResultBlock: tool_use_id={block.tool_use_id}, is_error={block.is_error}"
                                )
                    elif isinstance(message, ResultMessage):
                        session_id = message.session_id
                        # Update session_id_holder for hooks
                        session_id_holder["session_id"] = session_id
                        self.logger.info(
                            f"[{final_name}] ResultMessage: subtype={message.subtype}, "
                            f"session_id={message.session_id}, is_error={message.is_error}, "
                            f"num_turns={message.num_turns}, duration_ms={message.duration_ms}, "
                            f"cost_usd={message.total_cost_usd}, result={message.result}"
                        )
                        console.print(
                            Panel(
                                f"{message.result}",
                                title=f"Agent '{final_name}' (ResultMessage)",
                                border_style="green",
                            )
                        )
        except Exception as exc:
            raise RuntimeError(f"Claude agent initialization failed: {exc}") from exc

        if not session_id:
            raise RuntimeError("Failed to obtain session_id from Claude agent.")

        metadata = {
            "tool": tool,
            "type": agent_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "working_dir": str(AGENT_WORKING_DIRECTORY),
        }
        self._register_agent(final_name, session_id, metadata)

        ready_text = " ".join(transcript).strip()
        self.logger.info(f"Created agent '{final_name}' - session_id: {session_id}")

        return {
            "name": final_name,
            "session_id": session_id,
            "directory": str(agent_dir),
        }

    async def _prepare_operator_file(self, name: str, prompt: str) -> Path:
        """Prepare operator log file for task."""
        agent_dir = self._agent_directory(name)
        agent_dir.mkdir(parents=True, exist_ok=True)

        slug = await self._generate_operator_filename(prompt)
        filename = f"{slug}.md"
        operator_path = agent_dir / filename

        if operator_path.exists():
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            operator_path = agent_dir / f"{slug}-{timestamp}.md"

        header = textwrap.dedent(
            f"""
            # Operator Log · {name}

            **Task:** {prompt}
            **Created:** {datetime.now(timezone.utc).isoformat()}

            ## Status
            - Pending dispatch to agent.

            ---
            """
        ).strip()
        operator_path.write_text(header + "\n", encoding="utf-8")
        self._record_operator_file(name, operator_path)
        return operator_path

    def _run_agent_command_thread(
        self, agent_name: str, prompt: str, operator_path: Path
    ):
        """Run agent command in background thread."""
        try:
            asyncio.run(
                self._run_existing_agent_async(agent_name, prompt, operator_path)
            )
        except Exception as exc:
            self.logger.exception(f"Background command for '{agent_name}' failed")
            failure_note = textwrap.dedent(
                f"""
                ## Operator Update
                - **Status:** Failed to dispatch command.
                - **Error:** {exc}
                - **Timestamp:** {datetime.now(timezone.utc).isoformat()}
                """
            ).strip()
            with operator_path.open("a", encoding="utf-8") as fh:
                fh.write("\n" + failure_note + "\n")

    async def _run_existing_agent_async(
        self, agent_name: str, prompt: str, operator_path: Path
    ):
        """Run command on existing agent asynchronously."""
        agent = self._get_agent_by_name(agent_name)
        if not agent:
            raise RuntimeError(f"Agent '{agent_name}' not found in registry.")

        resume_session = agent.get("session_id")
        if not resume_session:
            raise RuntimeError(f"No session_id stored for agent '{agent_name}'.")

        system_prompt_text = self._render_prompt(
            "agentic_coder_system_prompt_system_prompt.md",
            OPERATOR_FILE=str(operator_path),
            WORKING_DIR=agent.get("working_dir", str(AGENT_WORKING_DIRECTORY)),
        )

        session_id_holder = {"session_id": resume_session or "unknown"}

        all_hook_types = [
            "PreToolUse",
            "PostToolUse",
            "Notification",
            "UserPromptSubmit",
            "Stop",
            "SubagentStop",
            "PreCompact",
            "SessionStart",
            "SessionEnd",
        ]

        hooks = {
            hook_type: [
                HookMatcher(
                    hooks=[
                        self._create_observability_hook(
                            agent_name, hook_type, session_id_holder
                        )
                    ]
                )
            ]
            for hook_type in all_hook_types
        }

        # Create browser tool if available
        mcp_servers = {}
        allowed_tools_list = [
            "Read",
            "Write",
            "Edit",
            "Bash",
            "Glob",
            "Grep",
            "Task",
            "WebFetch",
            "WebSearch",
            "BashOutput",
            "SlashCommand",
            "TodoWrite",
            "KillShell",
        ]
        disallowed_tools_list = ["NotebookEdit", "ExitPlanMode"]

        if self.browser_agent:
            browser_tool = self._create_browser_tool(agent_name)
            browser_server = create_sdk_mcp_server(
                name="browser", version="1.0.0", tools=[browser_tool]
            )
            mcp_servers["browser"] = browser_server
            allowed_tools_list.append("mcp__browser__browser_use")

        options = ClaudeAgentOptions(
            system_prompt={
                "type": "preset",
                "preset": "claude_code",
                "append": system_prompt_text,
            },
            model=DEFAULT_CLAUDE_MODEL,
            cwd=agent.get("working_dir", str(AGENT_WORKING_DIRECTORY)),
            permission_mode="bypassPermissions",
            resume=resume_session,
            setting_sources=["project"],
            hooks=hooks,
            mcp_servers=mcp_servers,
            allowed_tools=allowed_tools_list,
        )

        kickoff_note = textwrap.dedent(
            f"""
            ## Operator Update
            - **Status:** Task dispatched for execution.
            - **Prompt:** {prompt}
            - **Operator Log:** {operator_path}
            - **Timestamp:** {datetime.now(timezone.utc).isoformat()}
            """
        ).strip()

        with operator_path.open("a", encoding="utf-8") as fh:
            fh.write("\n" + kickoff_note + "\n")

        new_session_id: Optional[str] = None

        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)

            async for message in client.receive_response():
                # Log all message types
                if isinstance(message, UserMessage):
                    self.logger.info(f"[{agent_name}] UserMessage: {message.content}")
                elif isinstance(message, SystemMessage):
                    self.logger.info(
                        f"[{agent_name}] SystemMessage: subtype={message.subtype}, data={message.data}"
                    )
                elif isinstance(message, AssistantMessage):
                    self.logger.info(
                        f"[{agent_name}] AssistantMessage: model={message.model}, blocks={len(message.content)}"
                    )
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            self.logger.info(f"[{agent_name}] TextBlock: {block.text}")
                        elif isinstance(block, ThinkingBlock):
                            self.logger.info(
                                f"[{agent_name}] ThinkingBlock: {block.thinking}"
                            )
                        elif isinstance(block, ToolUseBlock):
                            self.logger.info(
                                f"[{agent_name}] ToolUseBlock: name={block.name}, id={block.id}, input={block.input}"
                            )
                        elif isinstance(block, ToolResultBlock):
                            self.logger.info(
                                f"[{agent_name}] ToolResultBlock: tool_use_id={block.tool_use_id}, is_error={block.is_error}"
                            )
                elif isinstance(message, ResultMessage):
                    new_session_id = message.session_id
                    # Update session_id_holder for hooks
                    session_id_holder["session_id"] = new_session_id
                    self.logger.info(
                        f"[{agent_name}] ResultMessage: subtype={message.subtype}, "
                        f"session_id={message.session_id}, is_error={message.is_error}, "
                        f"num_turns={message.num_turns}, duration_ms={message.duration_ms}, "
                        f"cost_usd={message.total_cost_usd}, result={message.result}"
                    )
                    console.print(
                        Panel(
                            f"{message.result}",
                            title=f"Agent '{agent_name}' (ResultMessage)",
                            border_style="green",
                        )
                    )

        if new_session_id and new_session_id != resume_session:
            with self.registry_lock:
                self.agent_registry["agents"][agent_name]["session_id"] = new_session_id
                self._save_agent_registry()

    # ------------------------------------------------------------------ #
    # Helper utilities
    # ------------------------------------------------------------------ #

    async def _generate_agent_name(self, existing_names: list[str]) -> str:
        """Generate unique agent name."""
        existing_display = (
            ", ".join(sorted(existing_names)) if existing_names else "none"
        )
        prompt_text = self._render_prompt(
            "agent_name_generator_user_prompt.md",
            EXISTING_NAMES=existing_display,
        )
        options = ClaudeAgentOptions(
            system_prompt="Return only the requested codename."
        )
        text = await self._collect_text_from_query(prompt_text, options)
        sanitized = "".join(ch for ch in text if ch.isalnum())
        return sanitized or f"Agent{datetime.now(timezone.utc).strftime('%H%M%S')}"

    async def _dedupe_agent_name(self, candidate: str) -> str:
        """Ensure agent name is unique."""
        name = candidate
        existing = self.agent_registry.get("agents", {})
        suffix = 1
        while name in existing:
            name = f"{candidate}{suffix}"
            suffix += 1
        return name

    async def _generate_operator_filename(self, prompt: str) -> str:
        """Generate operator log filename."""
        snippet = prompt.strip().replace("\n", " ")
        snippet = snippet[:160]
        prompt_text = self._render_prompt(
            "operator_filename_generator_user_prompt.md",
            PROMPT_SNIPPET=snippet,
        )
        options = ClaudeAgentOptions(system_prompt="Return only the slug requested.")
        text = await self._collect_text_from_query(prompt_text, options)
        slug = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in text.lower())
        slug = "-".join(filter(None, slug.split("-")))
        return slug or f"task-{datetime.now(timezone.utc).strftime('%H%M%S')}"

    async def _collect_text_from_query(
        self, prompt: str, options: ClaudeAgentOptions
    ) -> str:
        """Collect text response from query."""
        chunks: list[str] = []
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)
        return "".join(chunks).strip()

    def _record_operator_file(self, agent_name: str, operator_path: Path) -> None:
        """Record operator file in registry."""
        with self.registry_lock:
            agents = self.agent_registry.setdefault("agents", {})
            agent_entry = agents.setdefault(
                agent_name,
                {
                    "operator_files": [],
                    "session_id": None,
                    "tool": CLAUDE_CODE_TOOL,
                    "type": AGENTIC_CODING_TYPE,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "working_dir": str(AGENT_WORKING_DIRECTORY),
                },
            )
            files = agent_entry.setdefault("operator_files", [])
            path_str = str(operator_path)
            if path_str not in files:
                files.append(path_str)
            self._save_agent_registry()


# ================================================================
# OpenAIRealtimeVoiceAgent - Main orchestrator
# ================================================================


class WSLToolService:
    """Service layer exposing agentic capabilities for the Windows bridge."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("WSLToolService")
        self.browser_agent = GeminiBrowserAgent(logger=self.logger)
        self.agentic_coder = ClaudeCodeAgenticCoder(
            logger=self.logger, browser_agent=self.browser_agent
        )

    # ------------------------------------------------------------------ #
    # Agent registry helpers
    # ------------------------------------------------------------------ #

    def list_agents(self) -> Dict[str, Any]:
        """Return combined list of Claude and Gemini agents."""
        claude_result = self.agentic_coder.list_agents()
        claude_agents = claude_result.get("agents", [])

        browser_agents_list = []
        for name, data in sorted(
            self.browser_agent.agent_registry.get("agents", {}).items()
        ):
            browser_agents_list.append(
                {
                    "name": name,
                    "session_id": data.get("session_id"),
                    "tool": data.get("tool", GEMINI_TOOL),
                    "type": data.get("type", AGENTIC_BROWSERING_TYPE),
                    "created_at": data.get("created_at"),
                }
            )

        all_agents = claude_agents + browser_agents_list
        return {"ok": True, "agents": all_agents}

    def _create_browser_agent(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        browser_name = (
            agent_name
            or f"BrowserAgent_{datetime.now(timezone.utc).strftime('%H%M%S')}"
        )

        if self.browser_agent._get_agent_by_name(browser_name):
            return {
                "ok": False,
                "error": f"Browser agent '{browser_name}' already exists. Choose a different name.",
            }

        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "session_id": self.browser_agent.session_id,
        }
        self.browser_agent._register_agent(browser_name, metadata)

        return {
            "ok": True,
            "agent_name": browser_name,
            "session_id": self.browser_agent.session_id,
            "type": AGENTIC_BROWSERING_TYPE,
        }

    def create_agent(
        self,
        tool: str = CLAUDE_CODE_TOOL,
        type: str = AGENTIC_CODING_TYPE,
        agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if tool == GEMINI_TOOL and type == AGENTIC_BROWSERING_TYPE:
            return self._create_browser_agent(agent_name)
        if tool == CLAUDE_CODE_TOOL and type == AGENTIC_CODING_TYPE:
            return self.agentic_coder.create_agent(
                tool=tool, agent_type=type, agent_name=agent_name
            )
        return {
            "ok": False,
            "error": (
                f"Invalid tool/type combination: tool='{tool}', type='{type}'. "
                f"Valid combinations: ('{CLAUDE_CODE_TOOL}', '{AGENTIC_CODING_TYPE}') "
                f"or ('{GEMINI_TOOL}', '{AGENTIC_BROWSERING_TYPE}')"
            ),
        }

    def command_agent(self, agent_name: str, prompt: str) -> Dict[str, Any]:
        claude_agent = self.agentic_coder._get_agent_by_name(agent_name)
        browser_agent = self.browser_agent._get_agent_by_name(agent_name)

        if claude_agent:
            return self.agentic_coder.command_agent(
                agent_name=agent_name, prompt=prompt
            )
        if browser_agent:
            try:
                return self.browser_agent.execute_task(task=prompt)
            except Exception as exc:  # pragma: no cover - runtime safeguard
                self.logger.exception("Browser agent command failed")
                return {"ok": False, "error": f"Browser task failed: {exc}"}
        return {
            "ok": False,
            "error": f"Agent '{agent_name}' not found in either registry. Create it first.",
        }

    def check_agent_result(
        self, agent_name: str, operator_file_name: str
    ) -> Dict[str, Any]:
        return self.agentic_coder.check_agent_result(
            agent_name=agent_name, operator_file_name=operator_file_name
        )

    def delete_agent(self, agent_name: str) -> Dict[str, Any]:
        claude_agent = self.agentic_coder._get_agent_by_name(agent_name)
        browser_agent_data = self.browser_agent._get_agent_by_name(agent_name)

        if claude_agent:
            return self.agentic_coder.delete_agent(agent_name=agent_name)

        if browser_agent_data:
            warnings: list[str] = []
            with self.browser_agent.registry_lock:
                self.browser_agent.agent_registry.get("agents", {}).pop(
                    agent_name, None
                )
                try:
                    self.browser_agent._save_agent_registry()
                except Exception as exc:  # pragma: no cover - IO error
                    warnings.append(f"Failed to update registry: {exc}")

            agent_dir = self.browser_agent._agent_directory(agent_name)
            if agent_dir.exists():
                try:
                    shutil.rmtree(agent_dir)
                except Exception as exc:  # pragma: no cover - IO error
                    warnings.append(f"Failed to remove directory {agent_dir}: {exc}")

            payload: Dict[str, Any] = {"ok": True, "agent_name": agent_name}
            if warnings:
                payload["warnings"] = warnings
            return payload

        return {
            "ok": False,
            "error": f"Agent '{agent_name}' not found in either registry. Nothing to delete.",
        }

    def browser_use(
        self, task: str, url: Optional[str] = BROWSER_TOOL_STARTING_URL
    ) -> Dict[str, Any]:
        try:
            self.logger.info("Browser task: %s", task)
            return self.browser_agent.execute_task(task, url)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            self.logger.exception("Browser automation failed")
            return {"ok": False, "error": str(exc)}

    def open_file(self, file_path: str) -> Dict[str, Any]:
        try:
            full_path = AGENT_WORKING_DIRECTORY / file_path
            if not full_path.exists():
                return {"ok": False, "error": f"File not found: {file_path}"}

            media_extensions = {
                ".mp3",
                ".mp4",
                ".wav",
                ".m4a",
                ".aac",
                ".flac",
                ".ogg",
                ".mov",
                ".avi",
                ".mkv",
                ".webm",
                ".wmv",
                ".flv",
                ".m4v",
            }

            file_ext = full_path.suffix.lower()
            is_media = file_ext in media_extensions
            if is_media:
                command = f'open "{full_path}"'
                app_name = "default system application"
            else:
                command = f'code "{full_path}"'
                app_name = "VS Code"

            import subprocess

            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return {
                    "ok": True,
                    "file_path": str(full_path),
                    "opened_with": app_name,
                }
            error_msg = result.stderr or "Unknown error"
            return {"ok": False, "error": error_msg}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Command timed out after 5 seconds"}
        except Exception as exc:  # pragma: no cover - runtime safeguard
            self.logger.exception("Failed to open file")
            return {"ok": False, "error": str(exc)}

    def read_file(self, file_path: str) -> Dict[str, Any]:
        try:
            full_path = AGENT_WORKING_DIRECTORY / file_path
            if not full_path.exists():
                return {"ok": False, "error": f"File not found: {file_path}"}
            if full_path.is_dir():
                return {
                    "ok": False,
                    "error": f"Path is a directory, not a file: {file_path}",
                }
            try:
                content = full_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return {"ok": False, "error": f"File is not a text file: {file_path}"}
            return {
                "ok": True,
                "file_path": str(full_path),
                "content": content,
                "size": len(content),
                "lines": len(content.splitlines()),
            }
        except Exception as exc:  # pragma: no cover - runtime safeguard
            self.logger.exception("Failed to read file")
            return {"ok": False, "error": str(exc)}


# ================================================================
# Setup logging
# ================================================================


def setup_logging():
    """Setup logging to file only (no stdout)."""
    logger = logging.getLogger("BigThreeAgents")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    logger.propagate = False

    now = datetime.now()
    log_dir = Path(__file__).parent / "output_logs"
    log_dir.mkdir(exist_ok=True)
    log_filename = log_dir / f"{now.strftime('%Y-%m-%d_%H')}.log"
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger


# ================================================================
# Main entry point
# ================================================================


def main():
    """Entry point for CLI interactions."""
    parser = argparse.ArgumentParser(
        description=(
            "WSL tool service for the Big Three agents. Intended to be invoked "
            "from the Windows voice controller."
        )
    )
    parser.add_argument(
        "--bridge-call",
        choices=[
            "list_agents",
            "create_agent",
            "command_agent",
            "check_agent_result",
            "delete_agent",
            "browser_use",
            "open_file",
            "read_file",
        ],
        help="Execute a single bridge action and emit JSON to stdout.",
    )
    parser.add_argument(
        "--payload",
        help="Optional JSON payload. If omitted, payload is read from stdin.",
    )

    args = parser.parse_args()

    if not args.bridge_call:
        parser.print_help()
        return 0

    logger = setup_logging()
    payload_text = args.payload
    if payload_text is None:
        import sys

        payload_text = sys.stdin.read()
    payload_text = (payload_text or "").strip()
    if not payload_text:
        payload: Dict[str, Any] = {}
    else:
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON payload: {exc}")

    service = WSLToolService(logger=logger)
    handler = getattr(service, args.bridge_call)
    result = handler(**payload)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
