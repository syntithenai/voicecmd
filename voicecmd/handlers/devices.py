"""ESPHome plugs and lights, controlled by name through the kogan_switches panel.

The panel owns discovery; this handler polls its device list so new devices become
controllable by name without a restart.
"""

from __future__ import annotations

import difflib
import fnmatch
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..http import get_json, post_json
from ..intents import Intent, Reply

CONTROLLABLE = ("switch", "light")
FUZZY_MIN = 0.8
DIM_PCT = 30
FRESH_S = 2.0
# "set it to 10 percent" right after talking about a device means that device.
CONTEXT_S = 120.0
PRONOUNS = {"it", "that", "this", "them", "those"}

ON_OFF = r"(on|off)"
VERB = r"(?:turn|switch|power|shut|put)"
ALL_LIGHTS_RES = [
    re.compile(rf"^{VERB}\s+{ON_OFF}\s+(?:all\s+)?(?:of\s+)?(?:the\s+)?lights$"),
    re.compile(rf"^{VERB}\s+(?:all\s+)?(?:of\s+)?(?:the\s+)?lights\s+{ON_OFF}$"),
    re.compile(rf"^(?:all\s+)?(?:(?:of\s+)?the\s+)?lights\s+{ON_OFF}$"),
]
SET_RES = [
    re.compile(r"^(?:set|dim|turn|put|change|make|brighten)\s+(?:the\s+)?(.+?)\s+(?:to|at|up to|down to)\s+(\d{1,3})(?:\s*percent)?$"),
    re.compile(r"^(?:the\s+)?(.+?)\s+(?:to|at)\s+(\d{1,3})\s*percent$"),
]
DIM_RE = re.compile(r"^(?:dim|lower)\s+(?:the\s+)?(.+?)(?:\s+down)?$")
BRIGHTEN_RE = re.compile(r"^(?:brighten|turn\s+up)\s+(?:the\s+)?(.+?)(?:\s+(?:up|all the way|fully|to full))?$")
MAKE_RE = re.compile(
    r"^make\s+(?:the\s+)?(.+?)\s+(?:nice\s+and\s+|really\s+|fully\s+|a\s+bit\s+|a\s+little\s+)?"
    r"(bright|brighter|dim|dimmer|darker|cosy|cozy|low)$"
)
ON_OFF_RES = [
    re.compile(rf"^{VERB}\s+{ON_OFF}\s+(?:the\s+)?(.+)$"),
    re.compile(rf"^{VERB}\s+(?:the\s+)?(.+?)\s+{ON_OFF}$"),
    re.compile(rf"^(?:the\s+)?(.+?)\s+{ON_OFF}$"),
]
TOGGLE_RE = re.compile(r"^toggle\s+(?:the\s+)?(.+)$")
STATUS_RES = [
    re.compile(r"^(?:is|are)\s+(?:the\s+)?(.+?)\s+(?:switched\s+|turned\s+|still\s+)?(on|off|running)$"),
    re.compile(r"^(?:what's|what\s+is)\s+(?:the\s+)?(.+?)(?:'s)?\s+(?:status|state)$"),
]
POWER_RES = [
    re.compile(r"^how\s+much\s+(?:power|electricity|energy|juice)\s+(?:is|does)\s+(?:the\s+)?(.+?)\s+(?:using|use|drawing|draw|consuming|pulling)$"),
    re.compile(r"^(?:what's|what\s+is)\s+(?:the\s+)?(.+?)(?:'s)?\s+(?:power|power\s+usage|power\s+draw|wattage|usage)$"),
    re.compile(r"^how\s+many\s+watts\s+(?:is|does)\s+(?:the\s+)?(.+?)\s+(?:using|use|drawing|draw)$"),
]
LEADING_NOISE_RE = re.compile(r"^(?:the|my|our|a)\s+")
BARE_NAME_RE = re.compile(r"^(?:(?:the|um|uh|oh|i\s+mean)\s+)*(.+?)(?:\s+(?:one|please))?$")


def name_key(text: str) -> str:
    """'Tara's T.V.' -> 'taras tv'."""
    t = text.lower().replace("’", "'").replace("'", "")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\bt v\b", "tv", t)
    return re.sub(r"\s+", " ", t).strip()


def _variants(name: str) -> set[str]:
    key = name_key(name)
    out = {key}
    stripped = re.sub(r"\s+(?:light|lights|lamp|globe|plug|switch)$", "", key)
    if stripped and stripped != key:
        out.add(stripped)
    for k in list(out):
        if re.search(r"\btv\b", k):
            out.add(re.sub(r"\btv\b", "television", k))
    return out


def parse_aliases(spec: str) -> dict[str, str]:
    """'kettle=Jug; pool=Swimming Machine' -> {'kettle': 'Jug', 'pool': 'Swimming Machine'}."""
    out: dict[str, str] = {}
    for part in re.split(r"[;,]", spec or ""):
        if "=" in part:
            alias, target = (s.strip() for s in part.split("=", 1))
            if alias and target:
                out[name_key(alias)] = target
    return out


def parse_groups(spec: str) -> dict[str, tuple[str, list[tuple[str | None, str]]]]:
    """'office=light:*office*, *air filter*, *monitor*; garden=*pump*' -> {key: (name, [(domain, pattern)])}.

    Patterns are shell-style globs over device names, optionally limited to a domain, so devices
    that appear or get renamed later join the group automatically.
    """
    out: dict[str, tuple[str, list[tuple[str | None, str]]]] = {}
    for part in (spec or "").split(";"):
        if "=" not in part:
            continue
        name, members = (s.strip() for s in part.split("=", 1))
        rules: list[tuple[str | None, str]] = []
        for raw in members.split(","):
            raw = raw.strip()
            if not raw:
                continue
            domain, _, pattern = raw.rpartition(":")
            pattern = re.sub(r"\s+", " ", pattern.lower().replace("'", "").replace("’", "")).strip()
            rules.append((domain.strip().lower() or None, pattern))
        if name and rules:
            out[name_key(name)] = (name, rules)
    return out


def pct_to_brightness(pct: int) -> int:
    return max(1, min(255, int(pct * 255 / 100 + 0.5)))


def brightness_to_pct(brightness: int | None) -> int | None:
    return None if brightness is None else round(brightness * 100 / 255)


def _listed(names: list[str]) -> str:
    return ", ".join(names[:-1]) + f" and {names[-1]}" if len(names) > 1 else "".join(names)


def _watts(value: float) -> str:
    return f"{value:.1f} watts" if 0 < value < 10 else f"{round(value)} watts"


@dataclass
class Target:
    name: str
    ip: str
    entity_id: str
    domain: str
    state: str | None = None
    brightness: int | None = None
    online: bool = False
    power_w: float | None = None

    @property
    def is_on(self) -> bool:
        return (self.state or "").upper() == "ON"


def targets_from_panel(payload: dict) -> list[Target]:
    targets: list[Target] = []
    for dev in payload.get("devices") or []:
        ents = dev.get("entities") or []
        controls = [e for e in ents if e.get("domain") in CONTROLLABLE]
        if not controls:
            continue
        power = next((e.get("value") for e in ents
                      if e.get("domain") == "sensor" and re.search(r"\bpower$", str(e.get("name", "")), re.I)), None)
        friendly = ((dev.get("firmware") or {}).get("friendly") or "").strip()
        for ent in controls:
            name = friendly if friendly and len(controls) == 1 else (ent.get("name") or friendly or dev.get("name") or "")
            if not name:
                continue
            targets.append(Target(
                name=name,
                ip=dev.get("ip", ""),
                entity_id=ent.get("id", ""),
                domain=ent["domain"],
                state=ent.get("state") if isinstance(ent.get("state"), str) else None,
                brightness=ent.get("brightness"),
                online=bool(dev.get("online")),
                power_w=float(power) if isinstance(power, (int, float)) else None,
            ))
    return targets


class DeviceHandler:
    domain = "device"

    def __init__(
        self,
        url: str,
        refresh_s: float = 20.0,
        aliases: str | dict[str, str] = "",
        fetch: Callable[[], dict] | None = None,
        send: Callable[[dict], Any] | None = None,
        timeout_s: float = 3.0,
        groups: str = "",
    ):
        self.url = url.rstrip("/")
        self.refresh_s = refresh_s
        self.aliases = aliases if isinstance(aliases, dict) else parse_aliases(aliases)
        self.groups = parse_groups(groups)
        self.timeout_s = timeout_s
        self._fetch = fetch or (lambda: get_json(f"{self.url}/api/devices", timeout=self.timeout_s))
        self._send = send or (lambda body: post_json(f"{self.url}/api/command", body, timeout=self.timeout_s))
        self.targets: list[Target] = []
        self.reachable = False
        self.last_refresh = 0.0
        self.last_error = ""
        self._attempted = False
        self._last: tuple[str, float] | None = None
        self._pending: tuple[str, dict, float] | None = None
        self._lock = threading.Lock()
        self._listeners: list[Callable[[list[str]], None]] = []
        self._names: list[str] = []
        self._stop = threading.Event()

    # ---- registry -------------------------------------------------------------

    def on_names_changed(self, callback: Callable[[list[str]], None]) -> None:
        self._listeners.append(callback)
        if self._names:
            callback(list(self._names))

    def names(self) -> list[str]:
        return list(self._names)

    def group_names(self) -> list[str]:
        return [name for name, _ in self.groups.values()]

    def group_for(self, phrase: str) -> str | None:
        key = re.sub(r"^(?:the|my|our)\s+", "", name_key(phrase))
        key = re.sub(r"\s+(?:room|area)$", "", key)
        return key if key in self.groups else None

    def group_members(self, group_key: str) -> list[Target]:
        _, rules = self.groups[group_key]
        with self._lock:
            targets = list(self.targets)
        return [t for t in targets
                if any((domain is None or t.domain == domain) and fnmatch.fnmatchcase(name_key(t.name), pattern)
                       for domain, pattern in rules)]

    def refresh(self) -> bool:
        self._attempted = True
        try:
            targets = targets_from_panel(self._fetch() or {})
        except Exception as exc:
            if self.reachable:
                print(f"devices: panel unreachable at {self.url}: {exc}", flush=True)
            self.reachable, self.last_error = False, str(exc)
            return False
        names = sorted({t.name for t in targets}, key=str.lower)
        with self._lock:
            self.targets = targets
            changed = names != self._names
            self._names = names
        self.reachable, self.last_error, self.last_refresh = True, "", time.time()
        if changed:
            print(f"devices: {len(names)} controllable: {', '.join(names)}", flush=True)
            for cb in self._listeners:
                try:
                    cb(list(names))
                except Exception as exc:
                    print(f"devices: listener failed: {exc}", flush=True)
        return True

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.is_set():
                self.refresh()
                self._stop.wait(self.refresh_s)

        threading.Thread(target=loop, name="devices-refresh", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        with self._lock:
            targets = list(self.targets)
        return {
            "url": self.url,
            "reachable": self.reachable,
            "devices": len(targets),
            "online": sum(t.online for t in targets),
            "last_refresh": self.last_refresh,
            "error": self.last_error,
        }

    # ---- name resolution ------------------------------------------------------

    def resolve(self, phrase: str) -> list[Target]:
        """Best matching targets for a spoken name; more than one means ambiguous."""
        key = name_key(LEADING_NOISE_RE.sub("", phrase.strip().lower()))
        key = re.sub(r"^(?:the|my|our)\s+", "", key)
        if not key:
            return []
        with self._lock:
            targets = list(self.targets)
        by_name: dict[str, list[Target]] = {}
        for t in targets:
            by_name.setdefault(t.name, []).append(t)

        def named(names) -> list[Target]:
            return [t for n in dict.fromkeys(names) for t in by_name.get(n, [])]

        exact = [n for n in by_name if name_key(n) == key]
        if exact:
            return named(exact)
        variant = [n for n in by_name if key in _variants(n)]
        if variant:
            return named(variant)
        alias = self.aliases.get(key)
        if alias:
            hit = [n for n in by_name if name_key(n) == name_key(alias)]
            if hit:
                return named(hit)
        words = set(key.split())
        subset = [n for n in by_name if words <= set(name_key(n).split())]
        if subset:
            return named(subset)
        scored = sorted(
            ((max(difflib.SequenceMatcher(None, key, v).ratio() for v in _variants(n)), n) for n in by_name),
            reverse=True,
        )
        if scored and scored[0][0] >= FUZZY_MIN:
            best = scored[0][0]
            return named(n for score, n in scored if best - score < 0.02)
        return []

    # ---- parsing ----------------------------------------------------------------

    def _recent(self, slot: tuple | None) -> bool:
        return slot is not None and time.time() - slot[-1] < CONTEXT_S

    def _intent(self, phrase: str, action: str, **params) -> Intent | None:
        group = self.group_for(phrase)
        if group and action in ("on", "off", "status"):
            return Intent("device", action, {"group": self.groups[group][0]})
        if phrase.strip() in PRONOUNS:
            if not self._recent(self._last):
                return None
            phrase = self._last[0]
        hits = self.resolve(phrase)
        if action == "set":
            # "turn up the stereo" means volume, not a plug.
            hits = [t for t in hits if t.domain == "light"]
        if not hits:
            return None
        names = sorted(dict.fromkeys(t.name for t in hits), key=str.lower)
        if len(names) > 1:
            return Intent("device", "ambiguous", {"names": names, "want": action, **params})
        return Intent("device", action, {"name": names[0], **params})

    def parse(self, n: str, nw: str) -> Intent | None:
        if not self._attempted:
            self.refresh()  # one-shot CLI use, before the poller has run
        if not self.targets:
            return None
        for rx in ALL_LIGHTS_RES:
            m = rx.match(n)
            if m and any(t.domain == "light" for t in self.targets):
                return Intent("device", "all_lights", {"state": m.group(1)})
        for rx in POWER_RES:
            m = rx.match(n)
            if m:
                intent = self._intent(m.group(1), "power")
                if intent:
                    return intent
        for rx in STATUS_RES:
            m = rx.match(n)
            if m:
                intent = self._intent(m.group(1), "status")
                if intent:
                    return intent
        for rx in SET_RES:
            m = rx.match(n)
            if m:
                intent = self._intent(m.group(1), "set", percent=min(100, int(m.group(2))))
                if intent:
                    return intent
        m = MAKE_RE.match(n)
        if m:
            pct = 100 if m.group(2).startswith("bright") else DIM_PCT
            intent = self._intent(m.group(1), "set", percent=pct)
            if intent:
                return intent
        m = TOGGLE_RE.match(n)
        if m:
            intent = self._intent(m.group(1), "toggle")
            if intent:
                return intent
        for rx in ON_OFF_RES:
            m = rx.match(n)
            if m:
                state, phrase = (m.group(1), m.group(2)) if m.group(1) in ("on", "off") else (m.group(2), m.group(1))
                intent = self._intent(phrase, state)
                if intent:
                    return intent
        m = DIM_RE.match(n)
        if m:
            intent = self._intent(m.group(1), "set", percent=DIM_PCT)
            if intent:
                return intent
        m = BRIGHTEN_RE.match(n)
        if m:
            intent = self._intent(m.group(1), "set", percent=100)
            if intent:
                return intent
        if self._recent(self._pending):
            # Answer to "I found X and Y; which one?"
            want, params, _ = self._pending
            m = BARE_NAME_RE.match(n)
            hits = self.resolve(m.group(1)) if m else []
            if len({t.name for t in hits}) == 1:
                return Intent("device", want, {**params, "name": hits[0].name})
        return None

    # ---- execution --------------------------------------------------------------

    def _fresh(self) -> None:
        if time.time() - self.last_refresh > FRESH_S:
            self.refresh()

    def _command(self, t: Target, action: str, brightness: int | None = None) -> None:
        body: dict[str, Any] = {"ip": t.ip, "id": t.entity_id, "action": action}
        if brightness is not None:
            body["brightness"] = brightness
        self._send(body)
        with self._lock:
            if action == "turn_on":
                t.state = "ON"
                if brightness is not None:
                    t.brightness = brightness
            elif action == "turn_off":
                t.state = "OFF"
            elif action == "toggle":
                t.state = "OFF" if t.is_on else "ON"

    def execute(self, intent: Intent) -> Reply:
        p = intent.params
        if intent.action == "ambiguous":
            names = p.get("names") or []
            if p.get("want"):
                rest = {k: v for k, v in p.items() if k not in ("names", "want")}
                self._pending = (p["want"], rest, time.time())
            listed = ", ".join(names[:-1]) + f" and {names[-1]}" if len(names) > 1 else "".join(names)
            return Reply(f"I found {listed}; which one?", ok=False, intent=intent, expects_reply=True)
        self._pending = None
        self._fresh()
        if intent.action == "all_lights":
            return self._all_lights(intent, p.get("state", "off"))
        group = self.group_for(str(p.get("group") or p.get("name") or ""))
        if group and intent.action in ("on", "off", "status"):
            return self._group(intent, group)

        hits = self.resolve(str(p.get("name", "")))
        names = list(dict.fromkeys(t.name for t in hits))
        if not hits:
            return Reply(f"I don't know a device called {p.get('name', 'that')}.", ok=False, intent=intent)
        if len(names) > 1:
            return self.execute(Intent("device", "ambiguous", {"names": names}, intent.source))
        t = hits[0]
        self._last = (t.name, time.time())

        if intent.action == "status":
            if not t.online:
                return Reply(f"The {t.name} is offline.", intent=intent, data={"online": False})
            text = f"The {t.name} is {'on' if t.is_on else 'off'}"
            pct = brightness_to_pct(t.brightness) if t.domain == "light" and t.is_on else None
            if pct is not None:
                text += f" at {pct} percent"
            if t.power_w is not None and t.is_on:
                text += f", using {_watts(t.power_w)}"
            return Reply(text + ".", intent=intent, data={"state": t.state, "power_w": t.power_w})
        if intent.action == "power":
            if not t.online:
                return Reply(f"The {t.name} is offline.", intent=intent, data={"online": False})
            if t.power_w is None:
                return Reply(f"The {t.name} doesn't report its power.", ok=False, intent=intent)
            return Reply(f"The {t.name} is using {_watts(t.power_w)}.", intent=intent, data={"power_w": t.power_w})

        if not t.online:
            return Reply(f"The {t.name} is offline.", ok=False, intent=intent, data={"online": False})
        if intent.action in ("on", "off"):
            self._command(t, f"turn_{intent.action}")
            return Reply(f"{t.name} {intent.action}.", intent=intent)
        if intent.action == "toggle":
            self._command(t, "toggle")
            return Reply(f"{t.name} {'on' if t.is_on else 'off'}.", intent=intent)
        if intent.action == "set":
            pct = max(0, min(100, int(p.get("percent", 100))))
            if pct == 0:
                self._command(t, "turn_off")
                return Reply(f"{t.name} off.", intent=intent)
            if t.domain != "light":
                return Reply(f"The {t.name} can only be switched on or off.", ok=False, intent=intent)
            self._command(t, "turn_on", pct_to_brightness(pct))
            return Reply(f"{t.name} {pct} percent.", intent=intent)
        return Reply(f"I can't {intent.action} the {t.name}.", ok=False, intent=intent)

    def _group(self, intent: Intent, group_key: str) -> Reply:
        label = self.groups[group_key][0]
        members = self.group_members(group_key)
        if not members:
            return Reply(f"Nothing is in the {label} group yet.", ok=False, intent=intent)
        online = [t for t in members if t.online]
        offline = [t.name for t in members if not t.online]
        if intent.action == "status":
            on = [t.name for t in online if t.is_on]
            text = (f"In the {label}, {_listed(on)} {'is' if len(on) == 1 else 'are'} on."
                    if on else f"Everything in the {label} is off.")
        elif not online:
            return Reply(f"Everything in the {label} is offline.", ok=False, intent=intent)
        else:
            for t in online:
                self._command(t, f"turn_{intent.action}")
            text = f"{label.capitalize()} {intent.action}."
        if offline:
            text += f" {_listed(offline)} {'is' if len(offline) == 1 else 'are'} offline."
        return Reply(text, ok=bool(online), intent=intent, data={"members": [t.name for t in members]})

    def _all_lights(self, intent: Intent, state: str) -> Reply:
        with self._lock:
            lights = [t for t in self.targets if t.domain == "light"]
        if not lights:
            return Reply("I don't know any lights.", ok=False, intent=intent)
        online = [t for t in lights if t.online]
        for t in online:
            self._command(t, f"turn_{state}")
        offline = [t.name for t in lights if not t.online]
        text = f"All lights {state}." if len(online) != 1 else f"{online[0].name} {state}."
        if not online:
            text = "The lights are offline."
        elif offline:
            text += f" {', '.join(offline)} {'is' if len(offline) == 1 else 'are'} offline."
        return Reply(text, ok=bool(online), intent=intent)
