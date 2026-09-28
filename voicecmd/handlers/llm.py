"""LM Studio fallback: OpenAI-style tool calling mapped onto the same handler intents."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Callable

from ..http import post_json
from ..intents import Intent, Reply

SYSTEM_PROMPT = (
    "You are a voice assistant in a home. Replies are spoken aloud, so answer in one or two "
    "short plain sentences with no markdown, lists, emoji or URLs. "
    "Use a tool whenever the user wants music played or controlled, a timer or alarm, or "
    "anything about conditions outside (temperature, cold, hot, rain, humidity, garden soil): "
    "never invent weather readings, always call the weather tool. Convert durations exactly "
    "(a quarter of an hour is 900 seconds). Otherwise answer briefly from general knowledge. "
    "If you don't know, say so. Temperatures are Celsius. Current local time: {now}."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "music_play",
            "description": "Search the home music collection and play matching songs (title, artist, album or genre).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "e.g. 'dark side of the moon' or 'miles davis'"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "music_control",
            "description": "Control current music playback.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["pause", "resume", "next", "stop", "volume_up", "volume_down", "set_volume"]},
                    "level": {"type": "integer", "description": "0-100, only for set_volume"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "music_now_playing",
            "description": "Say what song is currently playing.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "timer_set",
            "description": "Start a countdown timer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {"type": "integer", "description": "total duration in seconds"},
                    "label": {"type": "string", "description": "optional name, e.g. 'pasta'"},
                },
                "required": ["seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "alarm_set",
            "description": "Set an alarm for a clock time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {"type": "string", "description": "24-hour HH:MM"},
                    "label": {"type": "string"},
                },
                "required": ["time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "timer_cancel",
            "description": "Cancel timers or alarms.",
            "parameters": {
                "type": "object",
                "properties": {"label": {"type": "string"}, "all": {"type": "boolean"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "timer_status",
            "description": "Say how long is left on timers and which alarms are set.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Read the home weather station (outdoor temperature, humidity, pressure, soil moisture, today's high/low).",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "enum": ["summary", "temperature", "humidity", "pressure", "soil", "today", "yesterday"]},
                },
            },
        },
    },
]


def tool_call_to_intent(name: str, args: dict) -> Intent | None:
    if name == "music_play":
        query = str(args.get("query") or "").strip()
        return Intent("music", "play_query", {"query": query}, source="llm") if query else None
    if name == "music_control":
        action = str(args.get("action") or "").strip()
        params = {"level": int(args["level"])} if args.get("level") is not None else {}
        return Intent("music", action, params, source="llm") if action else None
    if name == "music_now_playing":
        return Intent("music", "now_playing", source="llm")
    if name == "timer_set":
        seconds = int(args.get("seconds") or 0)
        return Intent("timer", "set", {"seconds": seconds, "label": str(args.get("label") or "")}, source="llm") if seconds > 0 else None
    if name == "alarm_set":
        return Intent("timer", "alarm", {"time": str(args.get("time") or ""), "label": str(args.get("label") or "")}, source="llm")
    if name == "timer_cancel":
        return Intent("timer", "cancel", {"label": str(args.get("label") or ""), "all": bool(args.get("all"))}, source="llm")
    if name == "timer_status":
        return Intent("timer", "status", source="llm")
    if name == "weather":
        return Intent("weather", str(args.get("topic") or "summary"), source="llm")
    return None


def spoken_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.S)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[*_#`>|]+", " ", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class LlmFallback:
    def __init__(self, url: str, model: str, timeout_s: float = 20.0, max_tokens: int = 400):
        self.url = url
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens

    def handle(self, text: str, execute: Callable[[Intent], Reply]) -> Reply:
        now = datetime.now().strftime("%A %d %B %Y, %H:%M")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT.format(now=now)},
                {"role": "user", "content": text},
            ],
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        started = time.monotonic()
        try:
            data = post_json(self.url, payload, timeout=self.timeout_s)
        except Exception as exc:
            print(f"llm: request failed: {exc}", flush=True)
            return Reply("Sorry, my thinking brain isn't available right now.", ok=False, data={"error": str(exc)})
        llm_ms = int((time.monotonic() - started) * 1000)
        message = ((data or {}).get("choices") or [{}])[0].get("message") or {}

        replies: list[Reply] = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            intent = tool_call_to_intent(str(fn.get("name") or ""), args if isinstance(args, dict) else {})
            if intent is not None:
                replies.append(execute(intent))
        if replies:
            first = replies[0]
            spoken = " ".join(r.text for r in replies if r.text and not r.silent)
            return Reply(spoken, ok=all(r.ok for r in replies), intent=first.intent,
                         expects_reply=replies[-1].expects_reply, data={"llm_ms": llm_ms, "tool_calls": len(replies)})

        answer = spoken_text(str(message.get("content") or ""))
        if not answer:
            return Reply("Sorry, I don't have an answer for that.", ok=False, data={"llm_ms": llm_ms})
        return Reply(answer, intent=Intent("llm", "answer", source="llm"),
                     expects_reply=answer.rstrip().endswith("?"), data={"llm_ms": llm_ms})
