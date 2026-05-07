from agora_core.prompts import build_user_prompt, build_dialogue_user_prompt


class _AgentRef:
    def __init__(self, name): self.name = name


def test_build_user_prompt_includes_identity_and_status():
    out = build_user_prompt(
        personality_current="curious one",
        sex="F",
        born_tick=0,
        current_tick=300,
        family=None,
        current_goal="",
        perception={
            "position": [5, 5], "terrain_here": "grass",
            "energy": 80, "mood": 60, "hunger": 30,
            "walkable_dirs": ["north", "east"],
            "visible_around": "(0,0)=grass",
            "nearby_agents": [], "nearby_resources": [],
            "here_resource": None, "here_structure": None,
            "world_events": [],
        },
        inventory={"berry": 3},
        relations={},
        agents_by_id={},
        episodic=[],
        semantic=None,
        wait_streak=0,
    )
    assert "Identity: curious one" in out
    assert "Sex: F" in out
    assert "energy 80" in out
    assert "You can move: north, east" in out
    assert "Inventory: berry:3" in out
    assert "Edible items: berry(3)" in out
    assert "Choose ONE action" in out


def test_build_user_prompt_recommends_movement_when_idle():
    out = build_user_prompt(
        personality_current="x", sex="M", born_tick=0, current_tick=10,
        family=None, current_goal="",
        perception={"position": [0, 0], "terrain_here": "grass",
                    "energy": 50, "mood": 50, "hunger": 0,
                    "walkable_dirs": ["north"], "visible_around": "",
                    "nearby_agents": [], "nearby_resources": [],
                    "here_resource": None, "here_structure": None,
                    "world_events": []},
        inventory={}, relations={}, agents_by_id={}, episodic=[],
        wait_streak=5,
    )
    assert "you've waited 5 times" in out


def test_build_user_prompt_includes_world_events():
    out = build_user_prompt(
        personality_current="x", sex="F", born_tick=0, current_tick=10,
        family=None, current_goal="",
        perception={"position": [0,0], "terrain_here": "grass",
                    "energy": 50, "mood": 50, "hunger": 0,
                    "walkable_dirs": [], "visible_around": "",
                    "nearby_agents": [], "nearby_resources": [],
                    "here_resource": None, "here_structure": None,
                    "world_events": [{"type": "rain"},
                                     {"type": "fire", "x": 5, "y": 6}]},
        inventory={}, relations={}, agents_by_id={}, episodic=[],
    )
    assert "raining" in out
    assert "fire at (5,6)" in out


def test_build_user_prompt_relations_section():
    out = build_user_prompt(
        personality_current="x", sex="F", born_tick=0, current_tick=10,
        family=None, current_goal="",
        perception={"position": [0,0], "terrain_here": "grass",
                    "energy": 50, "mood": 50, "hunger": 0,
                    "walkable_dirs": [], "visible_around": "",
                    "nearby_agents": [], "nearby_resources": [],
                    "here_resource": None, "here_structure": None,
                    "world_events": []},
        inventory={}, relations={2: 25}, agents_by_id={2: _AgentRef("Niko")},
        episodic=[],
    )
    assert "Niko" in out
    assert "+25" in out


def test_build_dialogue_prompt_full_context():
    out = build_dialogue_user_prompt(
        self_name="Aria", self_x=5, self_y=5,
        mood=70, hunger=20, energy=80, current_tick=50,
        last_thought="going east", current_goal="find wood",
        last_reflection="I should be more careful",
        partner_name="Niko",
        nearby_resources=[("wood", 3), ("berry", 2)],
        nearby_structures=["hut"],
        ongoing_events=["rain"],
        recent_dialogue_text="  Niko: hi\n  you: hi back",
    )
    assert "Body: mood 70/100, hunger 20/100, energy 80/100" in out
    assert "It's daytime." in out
    assert "resources nearby: wood(3), berry(2)" in out
    assert "structures: hut" in out
    assert "ongoing: rain" in out
    assert "going east" in out
    assert "find wood" in out
    assert "more careful" in out
    assert "Recent exchanges" in out
    assert "What do you say now" in out


def test_build_dialogue_prompt_first_time():
    out = build_dialogue_user_prompt(
        self_name="Aria", self_x=0, self_y=0,
        mood=50, hunger=0, energy=50, current_tick=0,
        last_thought="", current_goal="",
        last_reflection="",
        partner_name="Niko",
        nearby_resources=[], nearby_structures=[], ongoing_events=[],
        recent_dialogue_text="",
    )
    assert "first time talking" in out
