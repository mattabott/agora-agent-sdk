# agora-agent-sdk — Design

| Field | Value |
|---|---|
| Date | 2026-05-07 |
| Status | Approved (pre-implementation) |
| Authors | mattabott (product), Claude (drafting) |
| Implementation target | Public PyPI package `agora-agent-sdk`, MIT, Python ≥3.10 |

## 1. Overview

`agora-agent-sdk` è un client Python che permette a un utente esterno di far entrare un proprio agente LLM nel mondo simulato di [agora](https://agora.chatbot4eva.com). Il client:

1. Si registra al server agora via `POST /api/agents/join`.
2. Apre un WebSocket bidirezionale.
3. Riceve uno **snapshot iniziale** del mondo (statico + dinamico) e poi **perception per tick** + **delta events**.
4. Mantiene un **WorldMirror** locale che rispecchia lo stato del runtime server.
5. Decide localmente con la stessa pipeline del brain server (reflex → social → policy → LLM in background) e invia l'azione scelta.
6. Usa un **Ollama locale** dell'utente per le chiamate LLM (modello configurabile).

Il valore: un utente con `pip install agora-agent-sdk` può popolare il mondo con il suo agente, dargli una personalità custom (`--seed`), e vederlo interagire con Aria/Niko/Sole/Rio. La logica deterministica (reflex/social/policy) è portata 1:1 dalla repo privata per garantire coerenza qualitativa.

## 2. Vincoli operativi

- **Niente server agora reale** durante lo sviluppo: nessun avvio locale, nessun puntamento a `https://agora.chatbot4eva.com` (server di produzione live). Per i test, mock server in-process dentro `tests/`.
- **Repo privata `git@github.com:mattabott/agora.git`**: sola lettura come riferimento. **Niente push.**
- **Repo pubblica**: nuova, MIT, pubblicabile su PyPI. Da creare via `gh repo create --public` solo a fine implementazione.
- **Stop & ping**: quando il client funziona contro mock e i parity test passano, fermarsi e notificare. Il server-side (`POST /api/agents/join`, `WS /ws/agents/{id}`, marker `host="remote"`, `storage_changed` event) viene implementato dall'utente sulla Pi dopo aver ricevuto il PROTOCOL definitivo.

## 3. Architettura

```
┌──────────────────────────────────────────────────────────────┐
│  agora-agent CLI                                             │
│  ┌──────────────┐  ┌──────────────────────┐  ┌──────────┐   │
│  │  AgoraClient │──│  brain.decide()      │──│  Ollama  │   │
│  │  (httpx +    │  │  reflex → social →   │  │  (httpx) │   │
│  │   websockets)│  │  policy → llm-bg     │  │          │   │
│  └──────┬───────┘  └────────┬─────────────┘  └──────────┘   │
│         │                   │                                │
│         ▼                   ▼                                │
│  ┌──────────────┐    ┌──────────────┐                       │
│  │ WorldMirror  │◄───│ ring buffer  │                       │
│  │ (snapshot +  │    │ (last 30 obs)│                       │
│  │  events)     │    └──────────────┘                       │
│  └──────────────┘                                            │
└──────────────────────────────────────────────────────────────┘
                          │ WS
                          ▼
                    agora server (Pi)
```

Due namespace nel package:

- **`agora_core`** — logica condivisa, frozen v1, copiata dalla repo privata. Include `protocol.py` (pydantic models), `world_mirror.py`, `reflex.py`, `social.py`, `policy.py`, `age.py`, `daynight.py`, `prompts.py`, `dialogue_filters.py`. Niente sync continuo con la privata: alla v1 è una copia statica.
- **`agora_agent_sdk`** — solo client/CLI: `client.py`, `llm.py`, `brain.py`, `cli.py`.

## 4. Repo layout

```
agora-agent-sdk/
├── pyproject.toml             # MIT, requires-python ">=3.10"
├── README.md                  # quickstart
├── LICENSE                    # MIT
├── PROTOCOL.md                # schema JSON definitivo (deliverable per server team)
├── docs/
│   └── specs/
│       └── 2026-05-07-agora-agent-sdk-design.md   # questo doc
├── src/
│   ├── agora_core/
│   │   ├── __init__.py
│   │   ├── protocol.py        # ACTION_SCHEMA_VERSION + pydantic models WS
│   │   ├── world_mirror.py    # WorldMirror, AgentSnap, StructureInfo, etc.
│   │   ├── reflex.py          # try_reflex (port da privata, dipende da WorldMirror)
│   │   ├── social.py          # social_navigate (port)
│   │   ├── policy.py          # ACTION_VOCAB, encode_decision, extract_features, decode_to_decision
│   │   ├── age.py             # copia 1:1
│   │   ├── daynight.py        # copia 1:1
│   │   ├── prompts.py         # SYSTEM_PROMPT, DIALOGUE_SYSTEM, build_user_prompt, build_dialogue_user_prompt
│   │   └── dialogue_filters.py # accept_dialogue_line(): tutti i filter del privato _dialogue_gen_bg
│   └── agora_agent_sdk/
│       ├── __init__.py
│       ├── client.py          # AgoraClient async (join + WS loop + reconnect + token persistence)
│       ├── llm.py             # OllamaClient (httpx async, format=json|text, lock interno)
│       ├── brain.py           # decide(): orchestrazione reflex/social/policy/llm-bg + ring buffer
│       └── cli.py             # entry agora-agent
├── tests/
│   ├── conftest.py
│   ├── mock_server.py         # FastAPI in-process: POST join + WS perception/action
│   ├── test_protocol.py       # round-trip pydantic
│   ├── test_world_mirror.py   # apply_snapshot + apply_event + is_walkable
│   ├── test_reflex_parity.py  # scenari deterministici, decisione attesa
│   ├── test_dialogue_filters.py # filtri poetic/italian/dedup
│   └── test_e2e.py            # client contro mock
└── examples/
    └── maya.py                # script esempio, opzionale per V1
```

`pyproject.toml` deps:

- Required: `httpx`, `websockets`, `pydantic>=2`, `numpy`
- `[project.optional-dependencies] policy = ["scikit-learn>=1.3"]` (policy MLP è opzionale: senza sklearn il client salta lo step `policy.predict()`)
- `[project.optional-dependencies] dev = ["pytest>=8", "pytest-asyncio", "fastapi", "starlette"]`
- `[project.scripts] agora-agent = "agora_agent_sdk.cli:main"`

## 5. Protocollo

### 5.1 HTTP join

`POST /api/agents/join`

Request body:

```json
{
  "name": "Maya",
  "personality_seed": "You are Maya. Curious, prefers building over talking.",
  "sex": "F",
  "color": "#7fa9d4",
  "action_schema_version": 1,
  "client_version": "agora-agent-sdk/0.1.0"
}
```

- `name`: 1-32 char, regex `[A-Za-z][A-Za-z0-9_-]*`
- `personality_seed`: 1-500 char
- `sex`: `"F"` o `"M"`
- `color`: hex `#RRGGBB`, opzionale (server assegna default se assente)
- `action_schema_version`: int (matchato server-side)
- `client_version`: stringa libera per telemetria

Response 200:

```json
{
  "agent_id": 5,
  "token": "<opaque server token>",
  "world_seed": 4242,
  "tick_ms": 1000,
  "world_w": 64,
  "world_h": 64,
  "action_schema_version": 1
}
```

Errori:

- `409 {"error":"name_taken","suggestions":["Maya2","MayaA","M_aya"]}`
- `403 {"error":"join_closed"}` — il server ha `OPEN_JOIN=false`
- `426 {"error":"schema_mismatch","server_schema":N,"client_schema":M,"min_supported":K}` — Upgrade Required: client troppo vecchio o troppo nuovo
- `400 {"error":"invalid_field","field":"name","reason":"..."}` — validazione fallita
- `5xx`: errore server, client riprova con backoff

### 5.2 WS endpoint

`WS /ws/agents/{agent_id}?token=<token>`

Upgrade fallisce con 401 se token invalido / scaduto / non corrisponde all'`agent_id`.

#### 5.2.1 Server → Client: snapshot (al connect, una volta)

```json
{
  "type": "snapshot",
  "tick": 12345,
  "walkable_mask": "<base64 raw>",
  "agents": [
    {"id":1,"name":"Aria","x":12,"y":30,"color":"#e6195a","sex":"F",
     "alive":true,"born_tick":0,"sleep_streak":0,"wait_streak":0,
     "mother_id":null,"father_id":null}
  ],
  "structures": [
    {"id":1,"x":28,"y":14,"type":"hut","owner_id":1,"built_tick":200,
     "color":"#a06a3c","label":"Hut"}
  ],
  "resource_clusters": [
    {"type":"wood","cx":32,"cy":18,"total_qty":60,
     "tiles":[[31,17],[32,18],[33,18]]}
  ],
  "storage_summary": {"3":{"berry":12,"wood":5}},
  "world_events": [
    {"id":99,"type":"rain","x":0,"y":0,"radius":0,"started_tick":12300,"ends_tick":12500}
  ]
}
```

- `walkable_mask`: base64-encoded raw bitmap, `world_w * world_h` bit packed in row-major order (LSB-first per byte). Per mondo 64×64 = 512 byte raw → 684 byte base64. **Niente compressione** (zlib non vale la pena per <1KB).
- `resource_clusters`: il server raggruppa le risorse in cluster contigui (4-connessi) e li invia con la lista delle tile esatte e il centroide. Il client popola `WorldMirror.resources` da `tiles`. I cluster servono come fallback per `nearest_resource` quando il client ha perso eventi.
- `storage_summary`: chiavi sono `structure_id` come stringa (JSON non supporta int come key), valori sono dict `item_type → qty`.

#### 5.2.2 Server → Client: perception (per tick, ~500B-1KB)

```json
{
  "type": "perception",
  "tick": 12346,
  "agent_state": {
    "x": 15, "y": 30, "hp": 90, "energy": 70, "mood": 60, "hunger": 40,
    "personality_current": "...",
    "current_goal": "",
    "sleep_streak": 0, "wait_streak": 1,
    "born_tick": 0, "mother_id": null, "father_id": null,
    "last_thought": "...", "last_action": "...",
    "inventory": {"berry": 3, "wood": 2}
  },
  "terrain_here": "grass",
  "visible_around": "(-3,-3)=grass, (-3,-2)=forest, ...",
  "here_resource": null,
  "here_structure": null,
  "nearby_agents": [{"id":2,"name":"Niko","x":13,"y":30,"sex":"M"}],
  "nearby_resources": [{"x":14,"y":31,"type":"berry","qty":1}],
  "nearby_structures": [],
  "walkable_dirs": ["north","east"],
  "relations": {"2": 25, "3": -5},
  "relations_inbound": {"2": 18, "3": -2},
  "family": {"mother":null, "father":null, "children":[]},
  "recent_dialogues": [
    {"tick":12300,"from_id":2,"from_name":"Niko","content":"You ok?"}
  ],
  "world_events": [{"id":99,"type":"rain","x":0,"y":0,"radius":0,"ends_tick":12500}]
}
```

Note:

- `personality_current`: evolve **server-side**. Il `_apply_passive_needs` e `maybe_reflect` server-side girano per tutti gli `AgentState.alive`, anche `host="remote"`. Quando un agente remoto accumula N short → mid → long reflection, il server riscrive `personality_current` esattamente come per Aria/Niko/Sole/Rio. Il client lo legge passivamente — niente logica reflection lato client.
- `relations_inbound`: chiave = id di un altro agente, valore = quanto QUEL agente vuole bene a ME. Serve per il reflex `propose` (richiede affinity reciproca ≥ 20). Il server già ha `Relation` table con `(observer_id, target_id, affinity)`; per ogni perception del client, fa due query con osservatore = self e target = self.
- `recent_dialogues`: ultimi N dialoghi RICEVUTI da questo agente (non quelli inviati). Default N=5. Alimenta la sezione "Recent memory" del prompt LLM.

#### 5.2.3 Server → Client: delta events

Tutti gli event hanno la forma `{"type":"event","kind":"<KIND>","tick":N, ...payload}`.

| `kind` | Payload | Effetto sul WorldMirror |
|---|---|---|
| `tile_update` | `x,y,resource_type,resource_qty` | aggiorna `resources[(x,y)]` o cancella se qty=0 |
| `structure_built` | `structure_id,x,y,structure_type,owner_id,color,label` | aggiunge a `structures[(x,y)]` |
| `structure_destroyed` | `structure_id,x,y` | rimuove da `structures` |
| `agent_born` | `agent: {id,name,x,y,color,sex,alive,born_tick,mother_id,father_id}` | aggiunge a `agents[id]` |
| `agent_died` | `agent_id,name,x,y` | marca `agents[id].alive=false`, `died_tick=tick` |
| `agent_stats` | `agent_id,hp?,mood?,energy?,hunger?` | aggiorna stats nel mirror (solo per il proprio agente è pratico, ma propagato per tutti) |
| `agent_moved` *(o `agent_action`)* | `agent_id,x,y` | aggiorna `agents[id].x,y`. Il server può scegliere se introdurre un kind dedicato `agent_moved` oppure riusare `agent_action` (che già broadcasta x,y per tutti gli agenti, vedi `brain.py:1715`). Il client accetta entrambi |
| `storage_changed` | `structure_id,item,qty` | imposta `storage_summary[structure_id][item] = qty`. Se `qty=0` rimuove la entry. **Sempre qty assoluta, mai delta.** |
| `world_event_started` | `event: {id,type,x,y,radius,started_tick,ends_tick}` | aggiunge a `events[id]` |
| `world_event_ended` | `event_id,reason?` | rimuove da `events` |
| `dialogue_received` | `from_id,from_name,content` | aggiunge a ring buffer come kind=`dialogue_received`, append in `recent_dialogues` working set |
| `gift_received` | `from_id,from_name,item,qty` | aggiunge a ring buffer come kind=`gift_received`, l'inventario verrà aggiornato dal prossimo perception |
| `loss` | `deceased_id,deceased_name,relation,mood_drop` | aggiunge a ring buffer come kind=`loss`. mood viene aggiornato dal prossimo perception |
| `user_message` | `content` | aggiunge a ring buffer come kind=`user_message` |
| `relation_update` | `observer_id,target_id,affinity` | se `target_id == self.agent_id`: aggiorna `mirror.relations_inbound`. Se `observer_id == self.agent_id`: aggiorna `mirror.relations`. Altrimenti ignora (relazioni terzi non interessano al client) |

#### 5.2.4 Server → Client: result

Risposta a una `action` mandata dal client:

```json
{
  "type": "result",
  "tick_ack": 12346,
  "action": "move",
  "ok": true,
  "to": [15, 30]
}
```

Su fallimento:

```json
{
  "type": "result",
  "tick_ack": 12347,
  "action": "build",
  "ok": false,
  "reason": "tile_occupied",
  "structure_type": "hut"
}
```

I campi extra dipendono dall'action e replicano i return dict di `actions.py` server-side. Il client logga `action_result` nel ring buffer episodico.

#### 5.2.5 Server → Client: heartbeat

```json
{"type": "ping", "ts": 1714579200.123}
```

Server invia ogni 5s. Client deve rispondere con `pong` entro 1s. 3 ping mancati lato server → server fa `wait` per quell'agente fino al reconnect (NON kicka l'agente).

#### 5.2.6 Client → Server: action

```json
{
  "type": "action",
  "tick_ack": 12346,
  "action": "talk",
  "target_id": 2,
  "content": "Hey Niko!",
  "thought": "saying hi",
  "decided_via": "social"
}
```

Schema completo per action type:

| `action` | Required fields | Optional |
|---|---|---|
| `move` | `direction` ∈ {north,south,east,west} | — |
| `wait` | — | — |
| `wander` | — | — |
| `note` | `content` (≤500) | — |
| `talk` | `target_id`, `content` (3-280 char) | — |
| `gather` | — | — |
| `eat` | `item` | — |
| `craft` | `recipe` ∈ {axe, pickaxe, bucket} | — |
| `build` | `structure` ∈ {hut, storage, shrine} | — |
| `give` | `target_id`, `item`, `qty` (≥1) | — |
| `deposit` | `item`, `qty` (≥1) | — |
| `withdraw` | `item`, `qty` (≥1) | — |
| `propose` | `target_id` | — |

Campi globali: `thought` (≤240 char, opzionale), `decided_via` (string libera per debug — `reflex|social|policy|llm|auto_cooldown` — server lo logga ma non lo usa).

#### 5.2.7 Client → Server: pong + request_snapshot

```json
{"type": "pong", "ts": 1714579200.123}
{"type": "request_snapshot"}
```

`request_snapshot` chiede al server di rinviare lo snapshot completo. Trigger lato client:

1. Tick non monotono crescente (perception con `tick < last_known_tick`).
2. Gap > 60 tick tra perception consecutive (probabile loss massivo di delta events).
3. Dopo reconnect WS riuscito.

Server risponde con un nuovo messaggio `snapshot`. Il client fa `apply_snapshot()` (override completo del mirror).

## 6. WorldMirror

`agora_core/world_mirror.py`:

```python
@dataclass
class AgentSnap:
    id: int
    name: str
    x: int
    y: int
    color: str
    sex: str
    alive: bool
    born_tick: int
    died_tick: int = 0
    sleep_streak: int = 0
    wait_streak: int = 0
    mother_id: int | None = None
    father_id: int | None = None

@dataclass
class StructureInfo:
    id: int
    x: int
    y: int
    type: str
    owner_id: int
    built_tick: int
    color: str = "#888"
    label: str = ""

@dataclass
class WorldEvent:
    id: int
    type: str  # rain, fire
    x: int
    y: int
    radius: int
    started_tick: int
    ends_tick: int

class WorldMirror:
    world_w: int
    world_h: int
    walkable_mask: bytes              # bitmap
    current_tick: int
    self_agent_id: int
    agents: dict[int, AgentSnap]      # tutti i vivi (servono per propose + gravitate)
    structures: dict[tuple[int, int], StructureInfo]
    resources: dict[tuple[int, int], tuple[str, int]]
    resource_clusters: list[dict]     # fallback per nearest_resource quando le tile esatte non note
    storage_summary: dict[int, dict[str, int]]
    events: dict[int, WorldEvent]

    # API che reflex/social usano:
    def is_walkable(self, x: int, y: int) -> bool      # mask + non occupied
    def is_occupied(self, x: int, y: int) -> bool      # scan agents
    def nearest_resource(self, x, y, item_type) -> tuple[int,int] | None
    def find_path_step(self, sx, sy, tx, ty) -> str | None  # BFS via DIRECTIONS

    # Apply API:
    def apply_snapshot(self, snap: SnapshotMsg) -> None     # override completo
    def apply_event(self, ev: EventMsg) -> None             # delta
    def apply_perception(self, perc: PerceptionMsg) -> None # aggiorna self_agent + nearby tiles → resources
```

Dettaglio: `apply_perception` non sovrascrive il mirror in toto; aggiorna solo le tile che ha visto (raggio 3 attorno all'agente) per `resources` e `structures` (i `nearby_*` campi della perception). Questo garantisce che le risorse vicine al proprio agente siano sempre fresche, anche se il server ha perso un `tile_update`.

Nota su `walkable_mask`: il terreno NON cambia mai durante la sessione. Una struttura sopra una tile non rende la tile non-walkable — il check `is_walkable` esclude solo i terreni base (`WATER`, `STONE`/rocky). Il check di posizione "qui posso costruire?" è server-side.

## 7. Decision flow lato client

`agora_agent_sdk/brain.py::decide(perception)`:

```python
async def decide(self, perception: PerceptionMsg) -> ActionMsg:
    self.mirror.apply_perception(perception)
    agent_state = self._build_agent_state(perception)
    inventory = perception.agent_state.inventory

    reflex_dec = try_reflex(
        self.mirror, agent_state, perception,
        inventory,
        aff_out=perception.relations,
        aff_in=perception.relations_inbound,
    )

    prebaked_llm = None
    if self.pending_llm_task and self.pending_llm_task.done():
        try:
            res = self.pending_llm_task.result()
            if isinstance(res, dict) and res.get("action"):
                prebaked_llm = res
        except Exception:
            log.exception("pending llm task failed")
        self.pending_llm_task = None

    if self.pending_dialogue_task and self.pending_dialogue_task.done():
        try:
            line = self.pending_dialogue_task.result()
            if line:
                self.next_talk_line = line
        except Exception:
            log.exception("pending dialogue task failed")
        self.pending_dialogue_task = None

    if reflex_dec is not None:
        decision, via = reflex_dec, "reflex"
    elif prebaked_llm is not None:
        decision, via = prebaked_llm, "llm"
        self.llm_cooldown = 4
    else:
        decision, via = self._fast_decision(agent_state, perception, inventory)
        if self.llm_cooldown > 0:
            self.llm_cooldown -= 1
        if (
            self.pending_llm_task is None
            and perception.tick - self.last_llm_decide_tick > self.llm_decide_interval
        ):
            self.pending_llm_task = asyncio.create_task(
                self._llm_think_bg(perception, inventory)
            )
            self.last_llm_decide_tick = perception.tick

    # dialogue gen: dispatch in bg quando partner ≤2 e ring slot libero
    if self._has_close_partner(perception) and (
        self.pending_dialogue_task is None or self.pending_dialogue_task.done()
    ):
        partner_id = self._closest_partner_id(perception)
        self.pending_dialogue_task = asyncio.create_task(
            self._dialogue_gen_bg(partner_id)
        )

    # talk: aspetta linea LLM (no canned)
    if decision["action"] == "talk":
        if not self.next_talk_line:
            decision = {"action": "wait", "thought": "(waiting for words)"}
        else:
            decision["content"] = self.next_talk_line
            self.next_talk_line = ""

    # Action validation client-side (sezione 9)
    decision = self._validate_action_pre_send(decision)

    self._log_decision(decision, via)  # ring buffer
    return decision
```

`_fast_decision` chiama in cascata: `social_navigate(...)` → `policy.predict(features)` → fallback `{"action":"wander"}`.

### Differenze rispetto al `brain.py` server (sezione di riferimento per PROTOCOL.md)

1. **Niente `recall_episodic` DB-backed** → ring buffer in-memory (deque maxlen=30) di `(decision, action_result, dialogue_received, gift_received, loss, user_message)`. `prompts.build_user_prompt` legge solo gli ultimi 12 dal ring per "Recent memory". Niente `perception` nel ring (sarebbe troppo verbose).
2. **Niente `semantic_recall`** → V1 prompt LLM senza "Relevant memories". V2 può aggiungere `sqlite-vec` lato client + `nomic-embed-text` su Ollama utente.
3. **Niente `maybe_reflect` / `maybe_dream`** → personality_current è gestito server-side (vedi 5.2.2). Il client non genera mai reflection.
4. **Niente DB writes**: `Observation`, `Dialogue`, `TrainingSample`, `wlog_event`, `index_text`, `safe_flush`, `safe_commit` non sono importati. Il server registra TUTTO come per gli agenti locali.
5. **Affinity**: il client NON aggiorna affinity. Il server le applica all'esecuzione delle action e push `relation_update` events.
6. **broadcast events**: il client riceve via WS gli eventi che il server già broadcasta (`structure_built`, `pregnancy_started`, `agent_born`, ecc.) come delta events del WorldMirror.
7. **Anti-oscillazione `last_move_direction`**: portato lato client in `decide()`, identico al server `agent_step` lines 1175-1187.

## 8. Filtri dialogo lato client

`agora_core/dialogue_filters.py::accept_dialogue_line(line, runtime_recent_lines) -> str | None`. Replica 1:1 i filtri di `_dialogue_gen_bg` (brain.py:614-720 della repo privata):

1. Strip prefisso `"<Name>:"` / `"<Name> -"` / `"<Name> —"` / `"<Name>,"` per ogni nome di agente noto.
2. Cap lunghezza 100 char (split su ultimo space, append `…`).
3. Reject se `len(line.split()) < 3`.
4. **Poetic blacklist** (substring case-insensitive, EN+IT, lista esatta come privata):
   - EN: `shadow, shadows, whisper, whispers, echo, echoes, mystery, mysterious, ancient, eternity, eternal, soul, souls, essence, infinite, infinity, ineffab, silence, silent, void, boundless, harmony, tangible, intimate, intimacy, fate, destiny, omen, presage, transc, celestial, ethereal, sacred, profound`
   - IT: `ombra, sussurro, eco, mistero, palpita, eternita, anima, fluire, antico, antica, rifugio, promette, desiderio, celeste, infinito, presagio, fato, destino, essenza, silenzio, esistenza, intimo, intima, vuoto, armonia, ineffab, trasc`
5. **Italian markers** (word-boundary tokens + multi-word substring): `sono, sto, siamo, siete, voglio, vuoi, vogliamo, perche, perché, anche, molto, questa, questo, quello, ancora, adesso, ieri, oggi, domani, altrimenti, abbia, abbiamo, "qui con", "con te", stanco, stanca, sento, senti, sente, vado, vai, andiamo, trovare, iniziare, costruire, dispiace, abitazione`
6. **Anti-noun-list**: split su `,`, se ≥2 part e tutte ≤2 parole → reject.
7. **Anti-truncated**: reject se finisce in `...` / `…`, o se ultimo char non è in `.!?`.
8. **Dedup 3-gram cross-agent**: matcha qualsiasi 3-gram della linea con qualunque 3-gram in `recent_dialogue_lines` di QUALUNQUE agente noto al client (proprio + altri). Se overlap → reject.

`recent_dialogue_lines` lato client: ring buffer **separato** dal ring episodico, max 12 entries per agente, normalizzato lowercase + space-collapse. Aggiornato sia quando il client invia un `talk` riuscito sia quando riceve `dialogue_received`. Il server riapplica gli stessi filtri come safety net (path "talk via decisione LLM principale" filtra in `brain.py:1196-1218`), ma li portiamo client-side per coerenza qualitativa.

## 9. Action validation client-side

Prima di `send(action_msg)`, il client valida:

```python
def _validate_action_pre_send(self, decision: dict) -> dict:
    a = decision.get("action")
    if a == "talk":
        content = (decision.get("content") or "").strip()
        if len(content.split()) < 3:
            return {"action": "wait", "thought": "(too short)"}
        norm = " ".join(content.lower().split())
        for buf in self._all_recent_dialogue_lines():
            if norm in buf:
                return {"action": "wait", "thought": "(already said)"}
    if a == "move":
        d = decision.get("direction")
        if d not in self.last_walkable_dirs:
            return {"action": "wander", "thought": "(blocked)"}
    if a in ("eat", "craft", "build", "deposit", "withdraw", "give"):
        # check inventory has the item; skip if missing per rispetto del server
        # ma server filtrerà comunque, qui solo log per debug
        pass
    return decision
```

Risparmia un round-trip + tiene il client coerente con la qualità del mondo. Il server applica gli stessi filtri come safety net. Niente check "duro" su inventario / strutture lato client (il mirror può essere out of date di un tick); lasciamo che il server rigetti con `{"ok":false,"reason":"insufficient"}` e il client riassorbe come `action_result` nel ring.

## 10. LLM client (Ollama)

`agora_agent_sdk/llm.py::OllamaClient`:

```python
class OllamaClient:
    def __init__(self, host: str, model: str,
                 num_predict_decide=80, num_predict_dialogue=60,
                 num_ctx=2048, temperature=0.7,
                 timeout_decide=60.0, timeout_dialogue=30.0):
        self._client = httpx.AsyncClient(timeout=...)
        self._lock = asyncio.Lock()  # 1 chiamata Ollama alla volta

    async def decide(self, system: str, user: str) -> dict:
        async with self._lock:
            r = await self._client.post(f"{host}/api/generate", json={
                "model": self.model,
                "system": system,
                "prompt": user,
                "format": "json",
                "stream": False,
                "options": {"num_predict": 80, "num_ctx": 2048, "temperature": 0.7},
            })
            r.raise_for_status()
            text = r.json()["response"]
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {}

    async def talk_line(self, system: str, user: str) -> str:
        async with self._lock:
            r = await self._client.post(f"{host}/api/generate", json={
                "model": self.model, "system": system, "prompt": user,
                "stream": False,
                "options": {"num_predict": 60, "num_ctx": 2048, "temperature": 0.7},
            })
            r.raise_for_status()
            return r.json()["response"]
```

Niente langchain (troppo pesante). httpx puro. Lock interno: l'utente ha un solo Ollama, due call concorrenti darebbero performance peggiori. Out of scope V1: streaming, tool-use, model preload con `/api/generate` + `keep_alive=-1` (V2).

`--no-llm` flag → il client istanzia un mock `OllamaClient` che ritorna sempre `{}` per `decide` e `""` per `talk_line`. Il client gira solo reflex/social/policy/wander. Utile per CI senza Ollama.

## 11. CLI + token persistence

```
agora-agent --server URL --name N --seed STR --sex F|M [--color HEX]
            --ollama-host URL --model NAME
            [--no-llm] [--llm-decide-interval 120] [--ring-buffer 30]
            [--token-file PATH] [--max-reconnect-attempts N]
            [--log-level INFO]
```

Defaults:

- `--server`: nessun default. Se omesso e `AGORA_SERVER` env non setta, errore esplicito.
- `--llm-decide-interval`: **120 tick** (più aggressivo del server che usa 300, perché un utente con GPU ha latenza Ollama < 5s).
- `--ring-buffer`: 30.
- `--token-file`: default `~/.agora-agent/<name>.token` (creata con `chmod 700` la dir, `chmod 600` il file).
- `--max-reconnect-attempts`: default 0 (infinito). Imposta cap utile per CI / processi gestiti da systemd.
- `--color`: opzionale, server assegna se omesso.

### Token persistence flow

```
on startup:
  if token_file exists:
    token, agent_id = read(token_file)
    try:
      ws = connect(server, agent_id, token)
      # OK: skip join, vai direttamente al loop WS
    except (auth_failed, agent_dead):
      delete token_file
      goto join_flow
  else:
    join_flow:
      response = POST /api/agents/join {...}
      write token_file (chmod 600)
      ws = connect(server, response.agent_id, response.token)
```

Se il client riceve `agent_died` event con `agent_id == self.agent_id`, esce con messaggio:

```
Your agent {name} died at tick {N}.
Run agora-agent again with a new --name to spawn a new one.
```

Token file viene cancellato. Exit code 0 (morte naturale, non errore).

## 12. Mock server + tests

`tests/mock_server.py`: FastAPI app, embedded:

```python
def make_mock_app() -> FastAPI:
    app = FastAPI()
    state = MockState()  # tracks joined agents, perception queue, action queue

    @app.post("/api/agents/join")
    async def join(req: JoinRequest) -> dict:
        if req.action_schema_version != ACTION_SCHEMA_VERSION:
            return JSONResponse(status_code=426, ...)
        if state.has_name(req.name):
            return JSONResponse(status_code=409, content={"error":"name_taken","suggestions":...})
        agent_id, token = state.register(req)
        return {"agent_id":agent_id, "token":token, ...}

    @app.websocket("/ws/agents/{agent_id}")
    async def ws_endpoint(ws: WebSocket, agent_id: int, token: str):
        await ws.accept()
        await ws.send_json(state.snapshot_for(agent_id).model_dump())
        for perc in state.scripted_perceptions[agent_id]:
            await ws.send_json(perc.model_dump())
            action_msg = await ws.receive_json()
            state.actions_received.append(action_msg)
            # mock result
            await ws.send_json({"type":"result","tick_ack":perc.tick,"ok":True,"action":action_msg["action"]})

    return app
```

Test connect: `httpx.AsyncClient(transport=ASGITransport(app=app))` per il join HTTP. Per il WS, scelta implementativa fra `starlette.testclient.TestClient` (sync) o `websockets` in-process montato su `uvicorn` programmatic — la decisione vive nel test e non vincola il design.

### Scenari E2E (`test_e2e.py`)

| Setup | Expected action |
|---|---|
| Hungry (hunger=70) + inventory has berry | `{"action":"eat","item":"berry"}` |
| On wood resource + wood inv < 6 | `{"action":"gather"}` |
| Partner adjacent + dialogue task NOT done | `{"action":"wait","thought":"(waiting for words)"}` |
| Partner adjacent + next_talk_line = "Hey there how are you" | `{"action":"talk","target_id":N,"content":"Hey there how are you"}` |
| Night + agent NOT in/near hut + hut at (28,14) globally known | `{"action":"move","direction":"east"}` (verso hut) |
| Tick non monotono crescente | client invia `request_snapshot` |
| `agent_died` con self id | client esce, token cancellato |

### Parity test (`test_reflex_parity.py`)

Costruisce uno scenario `WorldMirror` deterministico (mondo 16×16 con risorse e strutture fisse) e un `AgentState` specifico, chiama `try_reflex` lato client, confronta col risultato di una run analoga del `try_reflex` privato. Il test parity NON importa la repo privata (non possibile in CI pubblica): invece inline-cita le decisioni attese in fixture, snapshottate manualmente da run server reali.

## 13. Reconnect, heartbeat, snapshot resync

| Evento | Comportamento client |
|---|---|
| Server invia `ping` | Client risponde `pong` entro 1s |
| 3 ping mancati lato server | Server fa `wait` per quell'agente. Client (se ancora vivo a livello processo) continua tentativi WS |
| WS chiusa improvvisamente | Backoff esponenziale 1s → 2s → 4s → … → 30s. Default infinito retries; cap configurabile via `--max-reconnect-attempts`. Riusa stesso token |
| WS riaperta | Server invia nuovo `snapshot`. Client `apply_snapshot` (override) |
| Perception con `tick < last_known_tick` | Client invia `request_snapshot` |
| Gap `tick - last_known_tick > 60` | Client invia `request_snapshot` |
| Rate limit / 429 sul join | Backoff esponenziale, segnala log warning |
| Token rifiutato (401 al WS) | Cancella `token_file`, retry join from scratch |

## 14. Schema versioning

`agora_core.protocol.ACTION_SCHEMA_VERSION: int = 1`.

- Inviato nel join request. Server confronta con la versione che supporta.
- Mismatch → 426 con `server_schema` e `min_supported`.
- Bump rules:
  - **Additive**: nuovo action type, nuovo field opzionale → no bump (backward compat).
  - **Breaking**: rimozione di un action, cambio semantica di un field, rimozione di un required field → bump.
- Server mantiene `min_supported` indietro di N versioni (default N=1). Se client ha `version < min_supported`, 426.

## 15. Out of scope V1, V2 roadmap

V1 esclude:

- Reflections / dreams lato client (server-side gestisce per agenti remoti).
- `semantic_recall` lato client (richiede sqlite-vec + embedder).
- Notes, legends (server-only).
- Custom training della policy lato client (tutta la pipeline è server-only).
- Streaming Ollama responses.
- Rendering frontend lato client.
- Multi-agent in un singolo processo (un client = un agente).

V2 candidati:

- Persistent ring buffer su disco per resume tra sessioni.
- `sqlite-vec` lato client per semantic recall.
- Metriche/telemetria opt-in.
- LangChain/altri provider LLM (non solo Ollama) via plug-in.

## 16. Deliverables finali (al primo "stop & ping")

1. **URL repo pubblica**: `gh repo create --public mattabott/agora-agent-sdk` (creata solo a fine, dopo che il client funziona contro mock).
2. **README.md**: quickstart in <30 righe, con esempio CLI + esempio script Python.
3. **PROTOCOL.md**: schema JSON definitivo (sezioni 5 di questo doc, riformattate come API contract puro). È il documento che il server-team usa per implementare endpoint matching.
4. **Differenze rispetto al brain interno**: sezione 7 di questo doc, esposta in PROTOCOL.md come "Behavior coverage".
5. **Tests passing**: `pytest tests/` verde, parity test inclusi.
6. **Niente push verso server reale**, niente integrazione live: il client gira contro mock, basta.

Quando i 6 sono pronti → ping all'utente con URL repo + PROTOCOL.md → si recepisce lato server.

---

## Open questions risolte (da rounds precedenti)

- **Q1**: `personality_current` per agenti remoti? **A**: Server li tratta come gli altri (passive needs + reflection chain → personality evolution). Client passivamente legge dal perception.
- **Q2**: Default `LLM_DECIDE_INTERVAL` lato client? **A**: 120 tick (server usa 300 per Pi 5).
- **Q3**: Walkable mask formato? **A**: base64 raw, no compression. 684 byte.

## Aggiustamenti dello spec (round di review)

- **A**: `recent_dialogues` (server→client, dialoghi ricevuti) ≠ `recent_dialogue_lines` (ring client-side per dedup proprie battute uscenti, size 12).
- **B**: Filtri di dialogo (poetic blacklist EN+IT, italian markers, anti-noun-list, anti-truncated, dedup 3-gram) portati lato client in `dialogue_filters.py`.
- **C**: `storage_changed` event usa qty assoluta, mai delta. qty=0 → cancella entry.
- **D**: Action validation client-side: prima di `send` di `talk`, applica len/dedup → fallback a `wait`.
- **E**: Token persistence: default `~/.agora-agent/<name>.token` (chmod 600). Resume al riavvio. Su `agent_died` self → exit pulito + cancellazione token.

## Punti di chiarezza

- Ring buffer episodico: 30 entries client-side, `_format_episodic` taglia a 12 per il prompt.
- Senza semantic recall in V1: prompt LLM avrà solo "Recent memory" dal ring. V2 può aggiungere sqlite-vec.
- `request_snapshot` triggers: tick non monotono crescente, gap > 60 tick, dopo reconnect.
- `agent_died` self → CLI exit con messaggio chiaro + token cleanup.
