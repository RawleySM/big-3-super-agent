#!/usr/bin/env python3
"""
Big Three Realtime Agents - Windows Voice Controller
====================================================

This module hosts the OpenAI Realtime API experience that needs
native Windows audio support. It delegates all Claude and Gemini
tooling to the companion ``big_3_WSL.py`` script that runs inside WSL.

Usage:
    # Auto-prompt mode (text only)
    uv run big_3_Windows.py --prompt "Create an agent and have it make changes"

    # Interactive text mode
    uv run big_3_Windows.py --input text --output text

    # Full voice interaction
    uv run big_3_Windows.py --input audio --output audio

    # Use mini model
    uv run big_3_Windows.py --mini --input text --output text

Arguments:
    --input {text,audio}   Input mode (default: text)
    --output {text,audio}  Output mode (default: text)
    --prompt TEXT          Auto-dispatch prompt (forces text mode)
    --mini                 Use mini realtime model
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
import base64
import logging
import threading
import argparse
import asyncio
import textwrap
import subprocess
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import websocket
import pyaudio
import numpy as np
from pynput import keyboard
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

# Rich table is imported lazily to avoid unnecessary dependency costs on Windows
try:
    from rich.table import Table
except ImportError:  # pragma: no cover - rich should already be available
    Table = None

# Environment setup
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    raise ImportError(
        "dotenv not found. Please install it with `pip install python-dotenv`"
    )

# The Claude, Gemini, and Playwright SDKs live in WSL. Tool calls are bridged via
# subprocess and therefore do not require those dependencies in this environment.

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


class GeminiBrowserAgent:  # pragma: no cover - retained for backward imports
    """Placeholder to aid IDEs; real implementation runs in WSL."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "GeminiBrowserAgent is not available in Windows mode. Use WSL bridge."
        )


class ClaudeCodeAgenticCoder:  # pragma: no cover - retained for backward imports
    """Placeholder to aid IDEs; real implementation runs in WSL."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "ClaudeCodeAgenticCoder is not available in Windows mode. Use WSL bridge."
        )


class WSLToolBridge:
    """Lightweight subprocess bridge to the WSL automation stack."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("WSLToolBridge")
        command = os.environ.get("WSL_TOOL_BRIDGE_CMD", "wsl python3")
        self.command_prefix = shlex.split(command)
        if not self.command_prefix:
            raise ValueError("WSL_TOOL_BRIDGE_CMD resolved to an empty command")

        script_override = os.environ.get("WSL_TOOL_BRIDGE_SCRIPT")
        if script_override:
            self.logger.debug("Using WSL tool bridge script override: %s", script_override)
            self.script_path = Path(script_override)
        else:
            self.script_path = Path(__file__).with_name("big_3_WSL.py")

        self.wsl_script_path = self._to_wsl_path(self.script_path)
        self.timeout = int(os.environ.get("WSL_TOOL_BRIDGE_TIMEOUT", "600"))

    def _invoke(self, action: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload or {}
        cmd = [*self.command_prefix, self.wsl_script_path, "--bridge-call", action]
        self.logger.debug("Invoking WSL bridge: %s", " ".join(cmd))
        completed = subprocess.run(
            cmd,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(
                f"WSL bridge call '{action}' failed with code {completed.returncode}: {stderr}"
            )
        output = completed.stdout.strip() or "{}"
        try:
            return json.loads(output)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"WSL bridge returned invalid JSON for action '{action}': {output}"
            ) from exc

    def _to_wsl_path(self, path: Path) -> str:
        """Convert a Windows or POSIX path to a POSIX path usable by WSL."""

        try:
            resolved = path.resolve()
        except FileNotFoundError:
            resolved = path

        as_posix = resolved.as_posix()
        if as_posix.startswith("/mnt/"):
            return as_posix

        if resolved.drive:
            drive = resolved.drive.rstrip(":").lower()
            remainder_parts = resolved.parts[1:]
            remainder = "/".join(remainder_parts)
            return f"/mnt/{drive}/{remainder}"

        return as_posix

    # ------------------------------------------------------------------ #
    # Tool call helpers
    # ------------------------------------------------------------------ #

    def list_agents(self) -> Dict[str, Any]:
        return self._invoke("list_agents")

    def create_agent(self, **kwargs) -> Dict[str, Any]:
        return self._invoke("create_agent", kwargs)

    def command_agent(self, **kwargs) -> Dict[str, Any]:
        return self._invoke("command_agent", kwargs)

    def check_agent_result(self, **kwargs) -> Dict[str, Any]:
        return self._invoke("check_agent_result", kwargs)

    def delete_agent(self, **kwargs) -> Dict[str, Any]:
        return self._invoke("delete_agent", kwargs)

    def browser_use(self, **kwargs) -> Dict[str, Any]:
        return self._invoke("browser_use", kwargs)

    def open_file(self, **kwargs) -> Dict[str, Any]:
        return self._invoke("open_file", kwargs)

    def read_file(self, **kwargs) -> Dict[str, Any]:
        return self._invoke("read_file", kwargs)

class OpenAIRealtimeVoiceAgent:
    """
    OpenAI Realtime Voice Agent with agentic coding and browser automation.

    Orchestrates voice interactions via OpenAI Realtime API and delegates
    tasks to Claude Code agents and Gemini browser automation.
    """

    def __init__(
        self,
        input_mode: str = "text",
        output_mode: str = "text",
        logger=None,
        realtime_model: str | None = None,
        startup_prompt: Optional[str] = None,
        auto_timeout: int = 60,
    ):
        """Initialize the unified voice agent."""
        self.input_mode = input_mode
        self.output_mode = output_mode
        self.logger = logger or logging.getLogger("OpenAIRealtimeVoiceAgent")
        self.realtime_model = realtime_model or REALTIME_MODEL_DEFAULT
        self.ws = None
        self.audio_queue = []
        self.running = False
        self.audio_interface = None
        self.audio_stream = None
        self.console = Console()

        # Validate OpenAI API key
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Auto mode settings
        self.startup_prompt = startup_prompt
        self.auto_mode = startup_prompt is not None
        self.awaiting_auto_close = False
        self.auto_timeout = auto_timeout
        self.auto_start_time = None

        # Tool call tracking
        self.default_output_modalities = (
            ["audio"] if self.output_mode == "audio" else ["text"]
        )
        self.pending_function_arguments: Dict[str, str] = {}
        self.completed_function_calls: set[str] = set()

        # Initialize WSL tool bridge
        self.tool_bridge = WSLToolBridge(logger=self.logger)

        # Build tool specs
        self.tool_specs = self._build_tool_specs()

        # Background threads
        self.background_threads: list[threading.Thread] = []

        # Audio pause control (shift+space toggle)
        self.audio_paused = False  # Manual pause via shift+space
        self.auto_paused_for_response = False  # Auto-pause during agent speech
        self.keyboard_listener = None
        self.shift_pressed = False  # Track shift key state

        # Token tracking and cost analysis
        self.response_count = 0
        self.token_summary_interval = 3  # Show summary every N responses
        self.cumulative_tokens = {
            "total": 0,
            "input": 0,
            "output": 0,
            "input_text": 0,
            "input_audio": 0,
            "output_text": 0,
            "output_audio": 0,
        }
        self.cumulative_cost_usd = 0.0

        # Latency tracking
        self.speech_stopped_timestamp = None
        self.first_audio_delta_timestamp = None

        self.logger.info(
            f"Initialized OpenAIRealtimeVoiceAgent - Input: {input_mode}, Output: {output_mode}"
        )
        self._log_tool_catalog()

    # ------------------------------------------------------------------ #
    # Logging and UI
    # ------------------------------------------------------------------ #

    def _log_panel(
        self,
        message: str,
        *,
        title: str = "Agent",
        style: str = "cyan",
        level: str = "info",
        expand: bool = True,
    ) -> None:
        """Log message to both console panel and file logger."""
        console.print(Panel(message, title=title, border_style=style, expand=expand))
        log_fn = getattr(self.logger, level, None)
        if log_fn:
            log_fn(message)

    def _log_tool_catalog(self) -> None:
        """Display available tools in a panel."""
        if not getattr(self, "tool_specs", None):
            return
        entries: list[str] = []
        for spec in self.tool_specs:
            name = spec.get("name", "unknown_tool")
            properties = spec.get("parameters", {}).get("properties", {}) or {}
            params = ", ".join(properties.keys())
            if params:
                entries.append(f"{name}({params})")
            else:
                entries.append(f"{name}()")

        syntax = Syntax(
            json.dumps(entries, indent=2, ensure_ascii=False),
            "json",
            theme="monokai",
            word_wrap=True,
        )
        console.print(
            Panel(
                syntax,
                title="Tool Catalog",
                border_style="cyan",
                expand=True,
            )
        )
        self.logger.info("Tool catalog loaded with %d tools", len(self.tool_specs))

    def _log_agent_roster_panel(self, agents_payload: list[Dict[str, Any]]) -> None:
        """Display agent roster in a table."""
        if not agents_payload:
            self._log_panel(
                "No registered agents yet. Use create_agent to spin one up.",
                title="Agent Roster",
                style="yellow",
            )
            return

        if Table is None:
            # Fallback when rich.table is unavailable
            roster_text = "\n".join(
                f"- {agent.get('name', '?')} ({agent.get('tool', '?')}/{agent.get('type', '?')})"
                for agent in agents_payload
            )
            self._log_panel(
                roster_text,
                title="Agent Roster",
                style="cyan",
            )
            return

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Name", style="bold")
        table.add_column("Session ID", overflow="fold")
        table.add_column("Type")
        table.add_column("Tool")
        table.add_column("Recent File", overflow="fold")
        # table.add_column("All Files", overflow="fold")

        for agent in agents_payload:
            files = agent.get("operator_files") or []

            # Extract relative paths from AGENT_WORKING_DIRECTORY
            relative_paths = []
            if files:
                for f in files:
                    try:
                        rel_path = Path(f).relative_to(AGENT_WORKING_DIRECTORY)
                        relative_paths.append(str(rel_path))
                    except ValueError:
                        # If path is not relative to AGENT_WORKING_DIRECTORY, use filename only
                        relative_paths.append(Path(f).name)

            # Recent operator file (most recent = last in list)
            recent_file = relative_paths[-1] if relative_paths else "—"

            # All operator files (relative paths)
            # all_files_display = "\n".join(relative_paths) if relative_paths else "—"

            table.add_row(
                agent.get("name", "?"),
                agent.get("session_id", "?"),
                agent.get("type", "?"),
                agent.get("tool", "?"),
                recent_file,
                # all_files_display,
            )

        console.print(Panel.fit(table, title="Agent Roster", border_style="cyan"))
        self.logger.debug("Listed %d agents", len(agents_payload))

    def _log_tool_request_panel(
        self, tool_name: str, call_id: str, arguments_str: str
    ) -> None:
        """Display tool request in a panel."""
        try:
            parsed_args = json.loads(arguments_str or "{}")
            syntax = Syntax(
                json.dumps(parsed_args, indent=2, ensure_ascii=False),
                "json",
                theme="monokai",
                word_wrap=True,
            )
        except Exception:
            syntax = arguments_str or "{}"

        console.print(
            Panel(
                syntax,
                title=f"Tool Request · {tool_name}",
                border_style="cyan",
                expand=True,
            )
        )
        self.logger.info(
            "Model requested tool '%s' (call_id=%s) with args=%s",
            tool_name,
            call_id,
            arguments_str,
        )

    def _play_soft_beep(self, frequency: int = 440):
        """Play a soft beep tone (non-blocking)."""
        try:
            duration = 0.12  # seconds
            sample_rate = 24000
            volume = 0.15  # Soft volume

            # Generate sine wave
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            beep = (np.sin(frequency * t * 2 * np.pi) * volume * 32767).astype(np.int16)

            # Play using pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16, channels=1, rate=sample_rate, output=True
            )
            stream.write(beep.tobytes())
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            self.logger.debug(f"Beep playback failed: {e}")

    def _on_key_press(self, key):
        """Handle key press to toggle audio pause (shift+space)."""
        try:
            # Track shift key state
            if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                self.shift_pressed = True
            # Toggle only when space is pressed while shift is held
            elif key == keyboard.Key.space and self.shift_pressed:
                self.audio_paused = not self.audio_paused
                status = "PAUSED" if self.audio_paused else "LIVE"
                emoji = "⏸️" if self.audio_paused else "🎤"
                color = "yellow" if self.audio_paused else "green"

                # Play beep: higher pitch when resuming, lower when pausing
                beep_freq = 520 if not self.audio_paused else 380
                threading.Thread(
                    target=self._play_soft_beep, args=(beep_freq,), daemon=True
                ).start()

                # Show status panel
                self._log_panel(
                    f"{emoji} {status}",
                    title="Audio Input",
                    style=color,
                    level="info",
                    expand=False,
                )
                self.logger.info(f"Audio input {status}")
        except Exception as e:
            self.logger.error(f"Error handling shift+space: {e}")

    def _on_key_release(self, key):
        """Handle key release to track shift key state."""
        try:
            if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                self.shift_pressed = False
        except Exception as e:
            self.logger.error(f"Error handling key release: {e}")

    def _start_keyboard_listener(self):
        """Start keyboard listener for shift+space toggle."""
        if self.input_mode != "audio":
            return

        try:
            self.keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press, on_release=self._on_key_release
            )
            self.keyboard_listener.daemon = True
            self.keyboard_listener.start()
            self.logger.info("Keyboard listener started (shift+space to pause/resume)")
        except Exception as e:
            self.logger.warning(f"Could not start keyboard listener: {e}")

    def _calculate_cost_from_usage(self, usage: Dict[str, Any]) -> float:
        """Calculate cost in USD from usage data based on current model."""
        # Official pricing per 1M tokens (text and audio priced separately)
        if "mini" in self.realtime_model.lower():
            # gpt-realtime-mini pricing
            text_input_price = 0.60
            text_output_price = 2.40
            audio_input_price = 10.00
            audio_output_price = 20.00
        else:
            # gpt-realtime (standard) pricing
            text_input_price = 4.00
            text_output_price = 16.00
            audio_input_price = 32.00
            audio_output_price = 64.00

        input_details = usage.get("input_token_details", {})
        output_details = usage.get("output_token_details", {})

        # Extract token counts (skip image tokens as requested)
        input_text_tokens = input_details.get("text_tokens", 0)
        input_audio_tokens = input_details.get("audio_tokens", 0)
        output_text_tokens = output_details.get("text_tokens", 0)
        output_audio_tokens = output_details.get("audio_tokens", 0)

        # Calculate costs separately for text and audio
        text_input_cost = input_text_tokens / 1_000_000 * text_input_price
        text_output_cost = output_text_tokens / 1_000_000 * text_output_price
        audio_input_cost = input_audio_tokens / 1_000_000 * audio_input_price
        audio_output_cost = output_audio_tokens / 1_000_000 * audio_output_price

        return text_input_cost + text_output_cost + audio_input_cost + audio_output_cost

    def _display_token_summary(self):
        """Display token usage and cost summary."""
        tokens = self.cumulative_tokens
        total_cost = self.cumulative_cost_usd

        # Calculate input and output costs separately
        if "mini" in self.realtime_model.lower():
            text_input_price = 0.60
            text_output_price = 2.40
            audio_input_price = 10.00
            audio_output_price = 20.00
        else:
            text_input_price = 4.00
            text_output_price = 16.00
            audio_input_price = 32.00
            audio_output_price = 64.00

        # Input cost breakdown
        input_text_cost = tokens["input_text"] / 1_000_000 * text_input_price
        input_audio_cost = tokens["input_audio"] / 1_000_000 * audio_input_price
        input_cost = input_text_cost + input_audio_cost

        # Output cost breakdown
        output_text_cost = tokens["output_text"] / 1_000_000 * text_output_price
        output_audio_cost = tokens["output_audio"] / 1_000_000 * audio_output_price
        output_cost = output_text_cost + output_audio_cost

        summary = (
            f"Responses: {self.response_count}\n"
            f"Total Tokens: {tokens['total']:,}\n"
            f"├─ Input: {tokens['input']:,} (text: {tokens['input_text']:,}, audio: {tokens['input_audio']:,})\n"
            f"└─ Output: {tokens['output']:,} (text: {tokens['output_text']:,}, audio: {tokens['output_audio']:,})\n"
            f"\n"
            f"Cost Breakdown:\n"
            f"├─ Input: ${input_cost:.4f}\n"
            f"├─ Output: ${output_cost:.4f}\n"
            f"└─ Total: ${total_cost:.4f} USD"
        )

        self._log_panel(
            summary,
            title="Token & Cost Summary",
            style="magenta",
            expand=False,
        )

    # ------------------------------------------------------------------ #
    # System prompt
    # ------------------------------------------------------------------ #

    def load_system_prompt(self) -> str:
        """Load system prompt for the orchestrator."""
        prompt_file = (
            PROMPTS_DIR / "super_agent" / "realtime_super_agent_system_prompt.md"
        )
        try:
            if prompt_file.exists():
                base_prompt = prompt_file.read_text(encoding="utf-8").strip()
                base_prompt = base_prompt.format(
                    AGENT_NAME=REALTIME_ORCH_AGENT_NAME, ENGINEER_NAME=ENGINEER_NAME
                )
            else:
                base_prompt = (
                    "You are a helpful voice assistant with advanced capabilities."
                )
        except Exception as e:
            self.logger.error(f"Error loading prompt file: {e}")
            base_prompt = (
                "You are a helpful voice assistant with advanced capabilities."
            )

        # Append active agent roster from WSL bridge
        try:
            roster = self.tool_bridge.list_agents().get("agents", [])
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.warning(f"Could not fetch agent roster: {exc}")
            roster = []

        if roster:
            roster_lines = ["\n# Active Agents"]
            for agent in roster:
                name = agent.get("name", "unknown")
                session_id = agent.get("session_id", "unknown")
                roster_lines.append(f"- {name} · session {session_id}")
            roster_block = "\n".join(roster_lines)
            base_prompt = f"{base_prompt}\n\n{roster_block}"

        return base_prompt

    # ------------------------------------------------------------------ #
    # Audio setup
    # ------------------------------------------------------------------ #

    def setup_audio(self):
        """Initialize PyAudio for audio input/output."""
        if self.input_mode != "audio" and self.output_mode != "audio":
            return

        self.logger.info("Setting up audio interface...")
        try:
            self.audio_interface = pyaudio.PyAudio()
            self.audio_stream = self.audio_interface.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK_SIZE,
            )
            self.logger.info("Audio interface ready")
        except Exception as e:
            self.logger.error(f"Failed to setup audio: {e}")
            raise

    def cleanup_audio(self):
        """Clean up audio resources."""
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.audio_interface:
            self.audio_interface.terminate()
        self.logger.info("Audio interface cleaned up")

    def base64_encode_audio(self, audio_bytes):
        """Encode audio bytes to base64."""
        return base64.b64encode(audio_bytes).decode("ascii")

    def base64_decode_audio(self, base64_str):
        """Decode base64 audio to bytes."""
        return base64.b64decode(base64_str)

    # ------------------------------------------------------------------ #
    # WebSocket handlers
    # ------------------------------------------------------------------ #

    def on_open(self, ws):
        """WebSocket connection opened."""
        self.logger.info("WebSocket connection established")
        self.running = True

        instructions = self.load_system_prompt()
        output_modalities = self.default_output_modalities

        session_config = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self.realtime_model,
                "output_modalities": output_modalities,
                "tool_choice": "auto",
                "tools": self.tool_specs,
                "instructions": instructions,
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": 24000,
                        },
                        "turn_detection": {"type": "semantic_vad"},
                        "transcription": {
                            "model": "gpt-4o-transcribe",
                        },
                    },
                    "output": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": 24000,
                        },
                        "voice": REALTIME_VOICE_CHOICE,
                    },
                },
            },
        }

        self.logger.info("Sending session configuration...")
        ws.send(json.dumps(session_config))

        if self.startup_prompt:
            self._log_panel(
                f"Auto prompt queued: {self.startup_prompt}\nTimeout: {self.auto_timeout}s",
                title="Auto Prompt",
                style="magenta",
            )
            # Record start time for timeout tracking
            self.auto_start_time = time.time()
            threading.Thread(
                target=self._dispatch_text_message,
                args=(self.startup_prompt,),
                daemon=True,
            ).start()
            self.awaiting_auto_close = True
        elif self.input_mode == "text":
            threading.Thread(target=self.text_input_loop, daemon=True).start()
        elif self.input_mode == "audio":
            threading.Thread(target=self.audio_input_loop, daemon=True).start()

    def on_message(self, ws, message):
        """Handle incoming server events."""
        try:
            event = json.loads(message)
            event_type = event.get("type", "unknown")

            if event_type == "error":
                self.logger.error(
                    f"ERROR EVENT RECEIVED: {json.dumps(event, indent=2)}"
                )

            # Enhanced conversation logging
            if event_type == "conversation.item.created":
                item = event.get("item", {})
                if item.get("type") == "message":
                    role = item.get("role")
                    text_parts = [
                        part.get("text", "")
                        for part in item.get("content", [])
                        if isinstance(part, dict)
                        and part.get("type") in {"input_text", "output_text"}
                    ]
                    message_text = "\n".join(filter(None, text_parts))
                    if message_text:
                        if role == "user":
                            self._log_panel(
                                message_text, title="User Input", style="blue"
                            )
                        elif role == "assistant":
                            self._log_panel(
                                message_text, title="Assistant", style="green"
                            )

            elif event_type == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript", "")
                if transcript:
                    self._log_panel(
                        transcript, title="User Input (Audio)", style="blue"
                    )

            elif event_type == "input_audio_buffer.speech_stopped":
                # Track when user stopped speaking for latency measurement
                self.speech_stopped_timestamp = time.time()
                self.first_audio_delta_timestamp = None  # Reset for next response
                self.logger.debug("User speech stopped, tracking latency")

            elif event_type == "response.output_text.done":
                final_text = event.get("text", "")
                if final_text:
                    self._log_panel(final_text, title="Assistant", style="green")

            elif event_type == "response.output_audio_transcript.done":
                transcript = event.get("transcript", "")
                if transcript:
                    # Calculate latency if we have both timestamps
                    title = "Assistant (Audio)"
                    if (
                        self.speech_stopped_timestamp
                        and self.first_audio_delta_timestamp
                    ):
                        latency = (
                            self.first_audio_delta_timestamp
                            - self.speech_stopped_timestamp
                        )
                        title = f"Assistant (Audio) ({latency:.3f}s)"

                    self._log_panel(transcript, title=title, style="green")

            elif event_type == "response.output_audio.delta":
                # Track first audio delta for latency measurement
                if self.first_audio_delta_timestamp is None:
                    self.first_audio_delta_timestamp = time.time()
                    if self.speech_stopped_timestamp:
                        latency = (
                            self.first_audio_delta_timestamp
                            - self.speech_stopped_timestamp
                        )
                        self.logger.debug(f"Voice latency: {latency:.3f}s")

                # Auto-pause audio input when agent starts speaking
                if not self.auto_paused_for_response and self.input_mode == "audio":
                    self.auto_paused_for_response = True
                    self.logger.debug("Auto-paused audio input (agent speaking)")

                audio_base64 = event.get("delta", "")
                if audio_base64 and self.output_mode == "audio" and self.audio_stream:
                    audio_bytes = self.base64_decode_audio(audio_base64)
                    self.audio_stream.write(audio_bytes)

            # Handle function calls
            if event_type == "response.function_call_arguments.delta":
                self._handle_function_call_delta(event)
            elif event_type == "response.done":
                self._handle_response_done(event)

                # Track token usage and cost
                response = event.get("response", {})
                usage = response.get("usage", {})

                if usage:
                    # Increment response count
                    self.response_count += 1

                    # Update cumulative tokens
                    self.cumulative_tokens["total"] += usage.get("total_tokens", 0)
                    self.cumulative_tokens["input"] += usage.get("input_tokens", 0)
                    self.cumulative_tokens["output"] += usage.get("output_tokens", 0)

                    input_details = usage.get("input_token_details", {})
                    output_details = usage.get("output_token_details", {})

                    self.cumulative_tokens["input_text"] += input_details.get(
                        "text_tokens", 0
                    )
                    self.cumulative_tokens["input_audio"] += input_details.get(
                        "audio_tokens", 0
                    )
                    self.cumulative_tokens["output_text"] += output_details.get(
                        "text_tokens", 0
                    )
                    self.cumulative_tokens["output_audio"] += output_details.get(
                        "audio_tokens", 0
                    )

                    # Calculate and accumulate cost
                    response_cost = self._calculate_cost_from_usage(usage)
                    self.cumulative_cost_usd += response_cost

                    # Log token usage every N responses (no UI pollution)
                    if self.response_count % self.token_summary_interval == 0:
                        tokens = self.cumulative_tokens
                        self.logger.info(
                            f"[Token Summary] Responses: {self.response_count}, "
                            f"Total: {tokens['total']:,}, "
                            f"Input: {tokens['input']:,} (text: {tokens['input_text']:,}, audio: {tokens['input_audio']:,}), "
                            f"Output: {tokens['output']:,} (text: {tokens['output_text']:,}, audio: {tokens['output_audio']:,}), "
                            f"Cost: ${self.cumulative_cost_usd:.4f} USD"
                        )

                # Resume audio input after agent finishes speaking
                if self.auto_paused_for_response and self.input_mode == "audio":
                    self.auto_paused_for_response = False
                    self.logger.debug("Resumed audio input (agent done speaking)")

                if self.auto_mode and self.awaiting_auto_close:
                    # Check elapsed time since auto-prompt started
                    elapsed = (
                        time.time() - self.auto_start_time
                        if self.auto_start_time
                        else 0
                    )

                    # Only consider closing if this response has NO function calls
                    response = event.get("response", {})
                    output_items = response.get("output", [])
                    has_function_calls = any(
                        item.get("type") == "function_call" for item in output_items
                    )

                    # Close if: text-only response AND timeout reached
                    if not has_function_calls and elapsed >= self.auto_timeout:
                        self.awaiting_auto_close = False

                        def _close():
                            self._log_panel(
                                f"Auto prompt complete after {elapsed:.1f}s (timeout: {self.auto_timeout}s); closing WebSocket.",
                                title="Auto Prompt",
                                style="magenta",
                            )
                            try:
                                ws.close()
                            except Exception as exc:
                                self._log_panel(
                                    f"Error closing WebSocket: {exc}",
                                    title="WebSocket Error",
                                    style="red",
                                    level="error",
                                )

                        threading.Timer(2.0, _close).start()
                    elif not has_function_calls:
                        # Text-only response but timeout not reached - log progress
                        self.logger.info(
                            f"Auto-prompt: Text response received. Waiting for timeout ({elapsed:.1f}s / {self.auto_timeout}s elapsed)"
                        )

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse message: {e}")
        except Exception as e:
            self.logger.error(f"Error handling message: {e}", exc_info=True)

    def on_error(self, ws, error):
        """WebSocket error handler."""
        self.logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """WebSocket connection closed."""
        self.logger.info(
            f"WebSocket connection closed: {close_status_code} - {close_msg}"
        )
        self.running = False

        # Check background threads
        for thread in self.background_threads:
            if thread.is_alive():
                self.logger.debug(f"Background task still running: {thread.name}")

    # ------------------------------------------------------------------ #
    # Input loops
    # ------------------------------------------------------------------ #

    def text_input_loop(self):
        """Handle text input from user."""
        self.logger.info(
            "Text input mode active. Type your messages (or 'quit' to exit):"
        )

        while self.running:
            try:
                user_input = input("\nYou: ")

                if user_input.lower() in ["quit", "exit", "q"]:
                    self.logger.info("User requested exit")
                    self.ws.close()
                    break

                if not user_input.strip():
                    continue

                self._dispatch_text_message(user_input)

            except EOFError:
                self.logger.info("EOF received, closing connection")
                break
            except Exception as e:
                self.logger.error(f"Error in text input loop: {e}")
                break

    def audio_input_loop(self):
        """Handle audio input from microphone."""
        self.logger.info(
            "Audio input mode active. Speak into your microphone (press SHIFT+SPACE to pause/resume):"
        )

        while self.running:
            try:
                # Skip audio capture when paused (manual or auto)
                if self.audio_paused or self.auto_paused_for_response:
                    time.sleep(0.1)
                    continue

                audio_data = self.audio_stream.read(
                    CHUNK_SIZE, exception_on_overflow=False
                )
                audio_base64 = self.base64_encode_audio(audio_data)
                event = {"type": "input_audio_buffer.append", "audio": audio_base64}
                self.ws.send(json.dumps(event))

            except Exception as e:
                self.logger.error(f"Error in audio input loop: {e}")
                break

    def _dispatch_text_message(self, text: str):
        """Send a text message and request a response."""
        if not self.ws:
            self._log_panel(
                "WebSocket unavailable; cannot dispatch text message.",
                title="Dispatch Error",
                style="red",
                level="error",
            )
            return

        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text,
                    }
                ],
            },
        }

        self.logger.info(f"Sending text message: {text}")
        self.ws.send(json.dumps(event))

        response_event = {
            "type": "response.create",
            "response": {"output_modalities": self.default_output_modalities},
        }
        self.ws.send(json.dumps(response_event))

    # ------------------------------------------------------------------ #
    # Function call handling
    # ------------------------------------------------------------------ #

    def _handle_function_call_delta(self, event: Dict[str, Any]):
        """Handle streaming function call arguments."""
        call_id = event.get("call_id")
        delta = event.get("delta", "")
        if not call_id or not delta:
            return

        self.pending_function_arguments[call_id] = (
            self.pending_function_arguments.get(call_id, "") + delta
        )

    def _handle_response_done(self, event: Dict[str, Any]):
        """Handle completed responses with function calls."""
        response = event.get("response", {})
        output_items = response.get("output", [])
        if not output_items:
            return

        for item in output_items:
            if item.get("type") != "function_call":
                continue

            call_id = item.get("call_id")
            if not call_id or call_id in self.completed_function_calls:
                continue

            tool_name = item.get("name") or "unknown"
            arguments_str = item.get(
                "arguments"
            ) or self.pending_function_arguments.pop(call_id, "")
            self._log_tool_request_panel(tool_name, call_id, arguments_str)
            self._execute_tool_call(
                call_id=call_id, tool_name=tool_name, arguments_str=arguments_str
            )

    def _execute_tool_call(
        self, call_id: str, tool_name: Optional[str], arguments_str: str
    ):
        """Execute a tool call and send the result back."""
        if not self.ws:
            self._log_panel(
                "WebSocket connection unavailable; cannot satisfy tool call.",
                title="Tool Error",
                style="red",
                level="error",
            )
            return

        parsed_args: Dict[str, Any] = {}
        if arguments_str:
            try:
                parsed_args = json.loads(arguments_str)
            except json.JSONDecodeError as exc:
                self._log_panel(
                    f"Failed to parse tool arguments: {exc}",
                    title="Tool Error",
                    style="red",
                    level="error",
                )
                payload = json.dumps(
                    {"ok": False, "error": f"Could not parse arguments: {exc}"}
                )
                self._send_function_output(call_id, payload)
                self.completed_function_calls.add(call_id)
                return

        handler_map = {
            "list_agents": self._tool_list_agents,
            "create_agent": self._tool_create_agent,
            "command_agent": self._tool_command_agent,
            "check_agent_result": self._tool_check_agent_result,
            "delete_agent": self._tool_delete_agent,
            "browser_use": self._tool_browser_use,
            "open_file": self._tool_open_file,
            "read_file": self._tool_read_file,
            "report_costs": self._tool_report_costs,
        }

        handler = handler_map.get(tool_name or "")
        if not handler:
            error_msg = f"Tool '{tool_name}' is not implemented on the server."
            self._log_panel(
                error_msg,
                title="Tool Error",
                style="red",
                level="error",
            )
            payload = json.dumps({"ok": False, "error": error_msg})
            self._send_function_output(call_id, payload)
            self.completed_function_calls.add(call_id)
            return

        try:
            result = handler(**parsed_args)
            payload = json.dumps(result)
        except Exception as exc:
            self._log_panel(
                f"Tool '{tool_name}' failed: {exc}",
                title="Tool Error",
                style="red",
                level="error",
            )
            self.logger.exception(f"Tool '{tool_name}' failed")
            payload = json.dumps({"ok": False, "error": f"Tool failed: {exc}"})

        self._send_function_output(call_id, payload)
        self.completed_function_calls.add(call_id)

        response_event = {
            "type": "response.create",
            "response": {"output_modalities": self.default_output_modalities},
        }
        self.ws.send(json.dumps(response_event))

    def _send_function_output(self, call_id: str, output_payload: str):
        """Send function output back to the model."""
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output_payload,
            },
        }
        self.ws.send(json.dumps(event))
        self.logger.debug(f"Emitted function_call_output for call_id={call_id}")

    # ------------------------------------------------------------------ #
    # Tool specifications
    # ------------------------------------------------------------------ #

    def _build_tool_specs(self) -> list[Dict[str, Any]]:
        """Build tool specifications for OpenAI Realtime API."""
        return [
            {
                "type": "function",
                "name": "list_agents",
                "description": "List all registered agents (both coding and browser automation) with session details and operator files.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "type": "function",
                "name": "create_agent",
                "description": (
                    "Create and register a new agent. Two tool/type combinations available:\n"
                    f"1. tool='{CLAUDE_CODE_TOOL}' + type='{AGENTIC_CODING_TYPE}' for software development tasks\n"
                    f"2. tool='{GEMINI_TOOL}' + type='{AGENTIC_BROWSERING_TYPE}' for browser automation tasks"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool": {
                            "type": "string",
                            "enum": [CLAUDE_CODE_TOOL, GEMINI_TOOL],
                            "description": f"Tool to use: '{CLAUDE_CODE_TOOL}' for coding agents, '{GEMINI_TOOL}' for browser automation",
                            "default": CLAUDE_CODE_TOOL,
                        },
                        "type": {
                            "type": "string",
                            "enum": [AGENTIC_CODING_TYPE, AGENTIC_BROWSERING_TYPE],
                            "description": f"Agent type: '{AGENTIC_CODING_TYPE}' for software development, '{AGENTIC_BROWSERING_TYPE}' for browser automation",
                            "default": AGENTIC_CODING_TYPE,
                        },
                        "agent_name": {
                            "type": "string",
                            "description": (
                                "Optional explicit codename for the agent. "
                                "If omitted, a unique name is generated."
                            ),
                        },
                    },
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "command_agent",
                "description": (
                    "Dispatch an asynchronous task to an existing agent (coding or browser automation). "
                    "For coding agents, returns the operator log path. For browser agents, executes the task directly."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "description": "Codename of the agent to command.",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Task prompt to send to the agent.",
                        },
                    },
                    "required": ["agent_name", "prompt"],
                },
            },
            {
                "type": "function",
                "name": "delete_agent",
                "description": (
                    "Remove a registered agent (coding or browser) and delete its working directory."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "description": "Codename of the agent to delete.",
                        },
                    },
                    "required": ["agent_name"],
                },
            },
            {
                "type": "function",
                "name": "check_agent_result",
                "description": (
                    "Read the operator status report for a given agent and file name."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "description": "Codename of the agent whose log to read.",
                        },
                        "operator_file_name": {
                            "type": "string",
                            "description": "Filename (including .md) inside the agent's directory.",
                        },
                    },
                    "required": ["agent_name", "operator_file_name"],
                },
            },
            {
                "type": "function",
                "name": "browser_use",
                "description": (
                    "Automate web browsing tasks using advanced AI. "
                    "Can navigate websites, search for information, interact with web pages, "
                    "extract data, and perform complex multi-step browsing tasks. "
                    "Provide a clear task description and optionally a starting URL."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": (
                                "A clear description of the browsing task to perform. "
                                "Be specific about what information to gather or actions to take."
                            ),
                        },
                        "url": {
                            "type": "string",
                            "description": (
                                "Optional starting URL. If not provided, will start from "
                                "a search engine or appropriate starting point."
                            ),
                        },
                    },
                    "required": ["task"],
                },
            },
            {
                "type": "function",
                "name": "open_file",
                "description": (
                    "Open a file in VS Code or the default system application. "
                    "Uses 'code' command for text/code files and 'open' command for media files (audio/video). "
                    "File path is automatically prefixed with the agent working directory."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": (
                                "Relative path to the file (e.g., 'frontend/src/App.vue' or 'backend/main.py'). "
                                "Will be prefixed with the working directory automatically."
                            ),
                        },
                    },
                    "required": ["file_path"],
                },
            },
            {
                "type": "function",
                "name": "read_file",
                "description": (
                    "Read and return the contents of a file from the working directory. "
                    "Useful for reviewing code, checking configurations, or gathering context before directing agents. "
                    "File path is automatically prefixed with the agent working directory."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": (
                                "Relative path to the file (e.g., 'backend/main.py' or 'specs/plan.md'). "
                                "Will be prefixed with the working directory automatically."
                            ),
                        },
                    },
                    "required": ["file_path"],
                },
            },
            {
                "type": "function",
                "name": "report_costs",
                "description": (
                    "Display current token usage and cost summary for this session. "
                    "Shows cumulative token counts (text/audio breakdown) and total cost in USD. "
                    "Use when the user asks about costs, usage, or spending."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        ]

    # ------------------------------------------------------------------ #
    # Tool implementations
    # ------------------------------------------------------------------ #

    def _tool_list_agents(self) -> Dict[str, Any]:
        """List all registered agents from both registries via WSL."""
        result = self.tool_bridge.list_agents()
        agents = result.get("agents", [])
        self._log_agent_roster_panel(agents)
        return result

    def _tool_create_agent(
        self,
        tool: str = CLAUDE_CODE_TOOL,
        type: str = AGENTIC_CODING_TYPE,
        agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new agent within WSL."""
        result = self.tool_bridge.create_agent(
            tool=tool, type=type, agent_name=agent_name
        )
        if result.get("ok"):
            name = result.get("agent_name") or agent_name or "(unnamed)"
            self._log_panel(
                f"Created agent '{name}' using tool={tool}, type={type}",
                title="Agent Created",
                style="green",
                expand=False,
            )
        else:
            self._log_panel(
                result.get("error", "Failed to create agent"),
                title="Agent Creation Failed",
                style="red",
                level="error",
            )
        return result

    def _tool_command_agent(self, agent_name: str, prompt: str) -> Dict[str, Any]:
        """Send a command to an agent through WSL."""
        result = self.tool_bridge.command_agent(agent_name=agent_name, prompt=prompt)
        if not result.get("ok"):
            self._log_panel(
                result.get("error", f"Failed to command {agent_name}"),
                title="Command Failed",
                style="red",
                level="error",
            )
        return result

    def _tool_check_agent_result(
        self, agent_name: str, operator_file_name: str
    ) -> Dict[str, Any]:
        """Check an agent result stored in WSL."""
        return self.tool_bridge.check_agent_result(
            agent_name=agent_name, operator_file_name=operator_file_name
        )

    def _tool_delete_agent(self, agent_name: str) -> Dict[str, Any]:
        """Delete an agent via the bridge."""
        result = self.tool_bridge.delete_agent(agent_name=agent_name)
        if result.get("ok"):
            self._log_panel(
                f"Deleted agent '{agent_name}'",
                title="Agent Deleted",
                style="yellow",
                expand=False,
            )
        else:
            self._log_panel(
                result.get("error", f"Failed to delete {agent_name}"),
                title="Delete Failed",
                style="red",
                level="error",
            )
        return result

    def _tool_browser_use(
        self, task: str, url: Optional[str] = BROWSER_TOOL_STARTING_URL
    ) -> Dict[str, Any]:
        """Delegate browser automation to WSL."""
        result = self.tool_bridge.browser_use(task=task, url=url)
        if result.get("ok"):
            summary = result.get("data") or result.get("summary") or "Task completed."
            screenshot_dir = result.get("screenshot_dir")
            message = summary
            if screenshot_dir:
                message += f"\nScreenshots: {screenshot_dir}"
            self._log_panel(message, title="Browser Task", style="blue")
        else:
            self._log_panel(
                result.get("error", "Browser automation failed"),
                title="Browser Task Failed",
                style="red",
                level="error",
            )
        return result

    def _tool_open_file(self, file_path: str) -> Dict[str, Any]:
        """Open a file using tooling in WSL."""
        result = self.tool_bridge.open_file(file_path=file_path)
        if not result.get("ok"):
            self._log_panel(
                result.get("error", f"Failed to open {file_path}"),
                title="Open File - Error",
                style="red",
                level="error",
            )
        return result

    def _tool_read_file(self, file_path: str) -> Dict[str, Any]:
        """Read a file through the WSL bridge."""
        result = self.tool_bridge.read_file(file_path=file_path)
        if result.get("ok"):
            self._log_panel(
                f"File: {result.get('file_path', file_path)}\nSize: {result.get('size', 0)} characters",
                title="Read File",
                style="cyan",
            )
        else:
            self._log_panel(
                result.get("error", f"Failed to read {file_path}"),
                title="Read File - Error",
                style="red",
                level="error",
            )
        return result

    def _tool_report_costs(self) -> Dict[str, Any]:
        """Display token usage and cost summary."""
        try:
            # Display the summary panel
            self._display_token_summary()

            # Return data for the tool result
            return {
                "ok": True,
                "responses": self.response_count,
                "total_tokens": self.cumulative_tokens["total"],
                "input_tokens": self.cumulative_tokens["input"],
                "output_tokens": self.cumulative_tokens["output"],
                "cost_usd": self.cumulative_cost_usd,
            }
        except Exception as exc:
            self._log_panel(
                f"Error generating cost report: {exc}",
                title="Report Costs - Error",
                style="red",
                level="error",
            )
            self.logger.exception("Failed to generate cost report")
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #

    def connect(self):
        """Connect to OpenAI Realtime API."""
        self.logger.info("Connecting to OpenAI Realtime API...")

        # Setup audio if needed
        if self.input_mode == "audio" or self.output_mode == "audio":
            self.setup_audio()

        # Start keyboard listener for audio pause control
        self._start_keyboard_listener()

        # Create WebSocket connection
        websocket_url = REALTIME_API_URL_TEMPLATE.format(model=self.realtime_model)
        self.ws = websocket.WebSocketApp(
            websocket_url,
            header=[f"Authorization: Bearer {OPENAI_API_KEY}"],
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )

        try:
            # Run WebSocket connection (blocking)
            self.ws.run_forever()
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            # Stop keyboard listener
            if self.keyboard_listener:
                self.keyboard_listener.stop()
                self.keyboard_listener = None

            if self.input_mode == "audio" or self.output_mode == "audio":
                self.cleanup_audio()
            self.logger.info("Connection closed")


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
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Big Three Realtime Agents - Unified agent system with voice, coding, and browser automation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        choices=["text", "audio"],
        default="text",
        help="Input mode: 'text' for typing, 'audio' for microphone",
    )
    parser.add_argument(
        "--output",
        choices=["text", "audio"],
        default="text",
        help="Output mode: 'text' for text responses, 'audio' for voice responses",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable voice mode (sets both input and output to audio)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Optional text prompt to auto-dispatch (forces text input/output).",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help=f"Use the mini realtime model ({REALTIME_MODEL_MINI}) instead of the default",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Auto-prompt mode timeout in seconds (default: 300). Keeps session alive for background agents to complete work.",
    )

    args = parser.parse_args()

    startup_prompt = None
    input_mode = args.input
    output_mode = args.output
    realtime_model = REALTIME_MODEL_MINI if args.mini else REALTIME_MODEL_DEFAULT

    # Voice flag overrides input/output settings
    if args.voice:
        input_mode = "audio"
        output_mode = "audio"

    # Prompt flag forces text mode (overrides --voice if both are set)
    if args.prompt:
        startup_prompt = args.prompt
        input_mode = "text"
        output_mode = "text"

    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Big Three Realtime Agents")
    logger.info("=" * 60)
    logger.info(f"Input: {input_mode}, Output: {output_mode}")
    logger.info(f"Realtime model: {realtime_model}")
    logger.info(f"Gemini model: {GEMINI_MODEL}")
    logger.info(f"Claude model: {DEFAULT_CLAUDE_MODEL}")
    logger.info(f"Agent working directory: {AGENT_WORKING_DIRECTORY}")
    if startup_prompt:
        logger.info(f"Auto prompt enabled: {startup_prompt}")

    config_message = (
        f"Input: {input_mode}\n"
        f"Output: {output_mode}\n"
        f"Realtime model: {realtime_model}\n"
        f"Gemini model: {GEMINI_MODEL}\n"
        f"Claude model: {DEFAULT_CLAUDE_MODEL}\n"
        f"Working dir: {AGENT_WORKING_DIRECTORY}"
    )
    console.print(
        Panel(config_message, title="Launch Configuration", border_style="cyan")
    )
    if startup_prompt:
        console.print(
            Panel(
                f"Auto prompt enabled:\n{startup_prompt}",
                title="Auto Prompt",
                border_style="magenta",
            )
        )

    try:
        agent = OpenAIRealtimeVoiceAgent(
            input_mode=input_mode,
            output_mode=output_mode,
            logger=logger,
            startup_prompt=startup_prompt,
            realtime_model=realtime_model,
            auto_timeout=args.timeout,
        )
        agent.connect()
    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
    except Exception as exc:
        logger.error(f"Fatal error: {exc}", exc_info=True)
        return 1

    logger.info("Agent terminated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
