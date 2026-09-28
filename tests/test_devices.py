import copy

import pytest

from voicecmd.handlers.devices import DeviceHandler, name_key, parse_aliases, pct_to_brightness
from voicecmd.handlers.llm import device_tools, tool_call_to_intent
from voicecmd.handlers.music import MusicHandler
from voicecmd.handlers.timers import TimerHandler
from voicecmd.intents import Intent
from voicecmd.normalize import normalize
from voicecmd.router import Router, SystemHandler, help_text
from voicecmd.stt import WhisperClient


def plug(ip, node, friendly, state="OFF", watts=0.0, online=True):
    return {
        "ip": ip, "name": node, "online": online, "firmware": {"friendly": friendly},
        "entities": [
            {"id": f"binary_sensor/{friendly} Status", "domain": "binary_sensor", "name": f"{friendly} Status", "state": state},
            {"id": f"sensor/{friendly} Power", "domain": "sensor", "name": f"{friendly} Power", "value": watts},
            {"id": f"sensor/{friendly} Daily Energy", "domain": "sensor", "name": f"{friendly} Daily Energy", "value": 0.1},
            {"id": f"switch/{friendly}", "domain": "switch", "name": friendly, "state": state, "value": state == "ON"},
        ],
    }


PANEL = {"devices": [
    plug("10.0.0.90", "coffee_machine", "Coffee Machine", online=False),
    plug("10.0.0.184", "jug", "Jug", state="ON", watts=1843.2),
    plug("10.0.0.101", "stereo", "Stereo"),
    plug("10.0.0.201", "taras_tv", "Taras TV", state="ON", watts=4.3),
    plug("10.0.0.160", "taras_washing_machine", "Taras Washing Machine"),
    plug("10.0.0.106", "swimming_machine", "Swimming Machine"),
    {"ip": "10.0.0.103", "name": "light_w_12", "online": True, "firmware": {"friendly": "Dining Room Light"},
     "entities": [{"id": "light/Dining Room Light", "domain": "light", "name": "Dining Room Light",
                   "state": "ON", "brightness": 255}]},
    {"ip": "10.0.0.50", "name": "weather_node", "online": True, "entities": [{"id": "sensor/Temp", "domain": "sensor"}]},
]}


class FakePanel:
    def __init__(self, payload):
        self.payload = copy.deepcopy(payload)
        self.sent = []
        self.down = False

    def fetch(self):
        if self.down:
            raise OSError("connection refused")
        return self.payload

    def send(self, body):
        self.sent.append(body)
        return {}


@pytest.fixture
def panel():
    return FakePanel(PANEL)


@pytest.fixture
def devices(panel):
    h = DeviceHandler("http://panel", aliases="kettle=Jug; pool=Swimming Machine", fetch=panel.fetch, send=panel.send)
    assert h.refresh()
    return h


def parse(h, text):
    return h.parse(normalize(text), normalize(text, numbers=False))


def test_only_controllable_devices_are_listed(devices):
    assert devices.names() == ["Coffee Machine", "Dining Room Light", "Jug", "Stereo", "Swimming Machine",
                               "Taras TV", "Taras Washing Machine"]


@pytest.mark.parametrize("spoken, name", [
    ("jug", "Jug"),                                # exact
    ("Tara's TV", "Taras TV"),                     # apostrophe
    ("taras t v", "Taras TV"),                     # spelled-out TV
    ("taras television", "Taras TV"),
    ("dining room", "Dining Room Light"),          # trailing "light" dropped
    ("kettle", "Jug"),                             # alias
    ("pool", "Swimming Machine"),
    ("washing machine", "Taras Washing Machine"),  # unique word subset
    ("the coffe machine", "Coffee Machine"),       # fuzzy
    ("stereos", "Stereo"),
])
def test_resolve(devices, spoken, name):
    assert [t.name for t in devices.resolve(spoken)] == [name]


def test_resolve_ambiguous_and_unknown(devices):
    assert {t.name for t in devices.resolve("machine")} == {"Coffee Machine", "Swimming Machine", "Taras Washing Machine"}
    assert devices.resolve("music") == []
    assert devices.resolve("volume") == []


@pytest.mark.parametrize("text, action, params", [
    ("turn off the jug", "off", {"name": "Jug"}),
    ("Switch on Tara's TV.", "on", {"name": "Taras TV"}),
    ("turn the stereo on", "on", {"name": "Stereo"}),
    ("kettle off", "off", {"name": "Jug"}),
    ("toggle the washing machine", "toggle", {"name": "Taras Washing Machine"}),
    ("set the dining room light to thirty percent", "set", {"name": "Dining Room Light", "percent": 30}),
    ("dining room to 50%", "set", {"name": "Dining Room Light", "percent": 50}),
    ("dim the dining room light", "set", {"name": "Dining Room Light", "percent": 30}),
    ("brighten the dining room", "set", {"name": "Dining Room Light", "percent": 100}),
    ("Make the dining room light nice and bright.", "set", {"name": "Dining Room Light", "percent": 100}),
    ("make the dining room cosy", "set", {"name": "Dining Room Light", "percent": 30}),
    ("turn off all the lights", "all_lights", {"state": "off"}),
    ("lights on", "all_lights", {"state": "on"}),
    ("is the jug on?", "status", {"name": "Jug"}),
    ("how much power is the washing machine using", "power", {"name": "Taras Washing Machine"}),
    ("what's the jug's power", "power", {"name": "Jug"}),
])
def test_parse(devices, text, action, params):
    intent = parse(devices, text)
    assert intent is not None and (intent.action, intent.params) == (action, params)


@pytest.mark.parametrize("text", [
    "turn off the music", "turn it off", "turn the volume up", "turn up the stereo", "set a timer for 5 minutes",
    "play some jazz", "turn everything off",
])
def test_parse_leaves_other_phrases_alone(devices, text):
    assert parse(devices, text) is None


def test_ambiguous_asks_which(devices):
    intent = parse(devices, "turn on the machine")
    assert intent.action == "ambiguous"
    reply = devices.execute(intent)
    assert reply.text == "I found Coffee Machine, Swimming Machine and Taras Washing Machine; which one?"
    assert reply.expects_reply


def test_execute_on_off_and_brightness(devices, panel):
    assert devices.execute(parse(devices, "turn off the jug")).text == "Jug off."
    assert panel.sent[-1] == {"ip": "10.0.0.184", "id": "switch/Jug", "action": "turn_off"}
    reply = devices.execute(parse(devices, "set the dining room light to 30 percent"))
    assert reply.text == "Dining Room Light 30 percent."
    assert panel.sent[-1] == {"ip": "10.0.0.103", "id": "light/Dining Room Light", "action": "turn_on", "brightness": 77}
    assert devices.execute(Intent("device", "set", {"name": "Dining Room Light", "percent": 0})).text == "Dining Room Light off."


def test_pct_to_brightness():
    assert [pct_to_brightness(p) for p in (1, 30, 50, 100, 150)] == [3, 77, 128, 255, 255]


def test_offline_and_switch_limits(devices, panel):
    reply = devices.execute(parse(devices, "turn on the coffee machine"))
    assert reply.text == "The Coffee Machine is offline." and not reply.ok
    reply = devices.execute(Intent("device", "set", {"name": "Stereo", "percent": 40}))
    assert "only be switched" in reply.text
    assert panel.sent == []


def test_status_and_power(devices):
    assert devices.execute(parse(devices, "is the jug on")).text == "The Jug is on, using 1843 watts."
    assert devices.execute(parse(devices, "is the stereo on")).text == "The Stereo is off."
    assert devices.execute(parse(devices, "is the dining room light on")).text == "The Dining Room Light is on at 100 percent."
    assert devices.execute(parse(devices, "how much power is taras tv using")).text == "The Taras TV is using 4.3 watts."


def test_all_lights_only_touches_lights(devices, panel):
    assert devices.execute(parse(devices, "turn off the lights")).text == "Dining Room Light off."
    assert [b["id"] for b in panel.sent] == ["light/Dining Room Light"]


def test_new_device_becomes_controllable(devices, panel):
    seen = []
    whisper = WhisperClient("http://127.0.0.1:0")
    devices.on_names_changed(seen.append)
    devices.on_names_changed(whisper.set_vocabulary)
    assert parse(devices, "turn on the fridge") is None

    panel.payload["devices"].append(plug("10.0.0.77", "taras_fridge", "Taras Fridge"))
    devices.refresh()
    assert parse(devices, "turn on the fridge").params == {"name": "Taras Fridge"}
    assert "Taras Fridge" in seen[-1]
    assert whisper.prompt.endswith("Taras Washing Machine.") and "Taras Fridge" in whisper.prompt
    enum = device_tools(devices.names())[0]["function"]["parameters"]["properties"]["name"]["enum"]
    assert "Taras Fridge" in enum


def test_pronoun_means_last_device(devices, panel):
    assert parse(devices, "set it to 10%") is None  # nothing talked about yet
    devices.execute(parse(devices, "is the dining room light on"))
    assert parse(devices, "Set it to 10%.").params == {"name": "Dining Room Light", "percent": 10}
    assert parse(devices, "turn it off").params == {"name": "Dining Room Light"}
    devices._last = ("Dining Room Light", 0.0)
    assert parse(devices, "turn it off") is None  # context expired


def test_bare_name_answers_which_one(devices, panel):
    reply = devices.execute(parse(devices, "turn on the machine"))
    assert reply.expects_reply
    intent = parse(devices, "the coffee machine")
    assert (intent.action, intent.params) == ("on", {"name": "Coffee Machine"})
    devices.execute(parse(devices, "jug off"))
    assert parse(devices, "the stereo") is None  # nothing pending any more


OFFICE = {"devices": [
    {"ip": "10.0.0.103", "name": "light_w_12", "online": True, "firmware": {"friendly": "Office Light"},
     "entities": [{"id": "light/Office Light", "domain": "light", "name": "Office Light", "state": "ON", "brightness": 255}]},
    plug("10.0.0.90", "coffee_machine", "Air Filter", state="ON"),
    plug("10.0.0.201", "taras_tv", "Office Desk", state="ON"),
    plug("10.0.0.184", "jug", "Jug", state="ON"),
    plug("10.0.0.101", "stereo", "AOC Monitor", online=False),
]}
OFFICE_GROUPS = "office=light:*office*, *air filter*, *monitor*"


@pytest.fixture
def office():
    p = FakePanel(OFFICE)
    h = DeviceHandler("http://panel", fetch=p.fetch, send=p.send, groups=OFFICE_GROUPS)
    h.refresh()
    return h, p


@pytest.mark.parametrize("text, action", [
    ("turn off the office", "off"), ("Turn on the office.", "on"), ("office off", "off"),
    ("switch the office room on", "on"), ("is the office on", "status"),
])
def test_group_phrases(office, text, action):
    h, _ = office
    intent = parse(h, text)
    assert (intent.action, intent.params) == (action, {"group": "office"})


def test_group_members_follow_names(office):
    h, p = office
    assert [t.name for t in h.group_members("office")] == ["Office Light", "Air Filter", "AOC Monitor"]
    p.payload["devices"][2] = plug("10.0.0.201", "taras_tv", "Doomsday Monitor")
    h.refresh()
    assert "Doomsday Monitor" in [t.name for t in h.group_members("office")]


def test_group_switches_online_members_only(office):
    h, p = office
    reply = h.execute(parse(h, "turn off the office"))
    assert reply.text == "Office off. AOC Monitor is offline."
    assert [b["id"] for b in p.sent] == ["light/Office Light", "switch/Air Filter"]
    assert all(b["action"] == "turn_off" for b in p.sent)


def test_group_status_and_single_device_still_works(office):
    h, p = office
    assert h.execute(parse(h, "is the office on")).text == \
        "In the office, Office Light and Air Filter are on. AOC Monitor is offline."
    assert parse(h, "turn off the office light").params == {"name": "Office Light"}
    assert parse(h, "dim the office").params == {"name": "Office Light", "percent": 30}
    assert h.execute(Intent("device", "off", {"name": "office"}, "llm")).text.startswith("Office off.")
    assert "office" in device_tools(h.names() + h.group_names())[0]["function"]["parameters"]["properties"]["name"]["enum"]


def test_panel_down_keeps_last_list(devices, panel):
    panel.down = True
    assert not devices.refresh()
    assert not devices.status()["reachable"]
    assert parse(devices, "jug off").params == {"name": "Jug"}


def test_device_tools_and_intents():
    assert device_tools([]) == []
    assert tool_call_to_intent("device_control", {"name": "Jug", "action": "off"}) == Intent("device", "off", {"name": "Jug"}, "llm")
    assert tool_call_to_intent("device_control", {"name": "Dining Room Light", "action": "on", "brightness": 20}) == \
        Intent("device", "set", {"name": "Dining Room Light", "percent": 20}, "llm")
    assert tool_call_to_intent("device_status", {"name": "Jug"}) == Intent("device", "status", {"name": "Jug"}, "llm")


def test_router_order_keeps_music_and_timers(devices, tmp_path):
    router = Router([TimerHandler(tmp_path / "timers.json"), devices, MusicHandler("http://127.0.0.1:0"),
                     SystemHandler(devices.names)])
    assert router.parse("turn off the music").domain == "music"
    assert router.parse("turn up the stereo") is None  # volume talk goes to the LLM, not the plug
    assert router.parse("set a timer for 5 minutes").domain == "timer"
    assert router.parse("jug off").domain == "device"
    assert "Jug" in router.execute(Intent("system", "help")).text


def test_help_text_and_aliases():
    assert help_text([]).endswith("Anything else I'll try to answer.")
    assert "I can also switch Jug and Stereo. Anything else" in help_text(["Jug", "Stereo"])
    assert parse_aliases("kettle = Jug, pool=Swimming Machine; bad") == {"kettle": "Jug", "pool": "Swimming Machine"}
    assert name_key("Tara's T.V.") == "taras tv"
