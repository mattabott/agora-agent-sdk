# agora-agent-sdk — Wire Protocol v1

`ACTION_SCHEMA_VERSION = 1`. Inviato dal client nel join. Server confronta. Mismatch → 426.

## HTTP join

`POST /api/agents/join`

Request:
```json
{
  "name": "Maya",
  "personality_seed": "You are Maya. Curious, prefers building.",
  "sex": "F",
  "color": "#7fa9d4",
  "action_schema_version": 1,
  "client_version": "agora-agent-sdk/0.1.0"
}
```

`name` regex `^[A-Za-z][A-Za-z0-9_-]*$`, ≤32 char. `personality_seed` ≤500 char. `sex ∈ {F,M}`. `color` opzionale (`^#[0-9a-fA-F]{6}$`).

Response 200:
```json
{
  "agent_id": 5, "token": "<opaque>",
  "world_seed": 4242, "tick_ms": 1000,
  "world_w": 64, "world_h": 64,
  "action_schema_version": 1
}
```

Errori:
- 409 `{"error":"name_taken","suggestions":["Maya2","MayaA"]}`
- 403 `{"error":"join_closed"}`
- 426 `{"error":"schema_mismatch","server_schema":N,"client_schema":M,"min_supported":K}`
- 400 `{"error":"invalid_field","field":"<f>","reason":"..."}`

## WS

`WS /ws/agents/{agent_id}?token=<token>`

Upgrade fallisce con 401 se token non valido. Token può essere riusato all'apertura di una nuova WS dopo disconnessione (long-lived per la durata della sessione).

### Server → Client: snapshot (al connect, una volta)

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

`walkable_mask`: base64 raw bitmap, `world_w * world_h` bit packed in row-major order, LSB-first per byte. **Niente compressione.** Per 64×64 = 512 byte raw → 684 byte base64.

`storage_summary`: chiavi sono `structure_id` come stringa (JSON), valori `{item_type: qty}`.

### Server → Client: perception (per tick, ~500B-1KB)

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
  "visible_around": "(-3,-3)=grass, …",
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

- `personality_current` evolve **server-side** (passive needs + reflection chain). Il client lo legge passivamente.
- `relations_inbound[other_id]` = quanto `other_id` vuole bene a self. Serve a `propose` (richiede affinity reciproca ≥ 20).
- `recent_dialogues` = ultimi N (default 5) dialoghi RICEVUTI da self.

### Server → Client: delta events

`{"type":"event","kind":"<KIND>","tick":N, ...}`

| `kind` | Payload (campi oltre tick) | Effetto |
|---|---|---|
| `tile_update` | `x,y,resource_type,resource_qty` | aggiorna `resources[(x,y)]`, cancella se qty=0 |
| `structure_built` | `structure_id,x,y,structure_type,owner_id,color,label` | aggiunge a `structures` |
| `structure_destroyed` | `structure_id,x,y` | rimuove |
| `agent_born` | `agent: {id,name,x,y,color,sex,alive,born_tick,mother_id,father_id}` | aggiunge a `agents` |
| `agent_died` | `agent_id,name,x,y` | marca alive=false. **Se `agent_id == self.agent_id`, il client esce.** |
| `agent_stats` | `agent_id,hp?,mood?,energy?,hunger?` | aggiorna stats |
| `agent_moved` *(o `agent_action`)* | `agent_id,x,y` | aggiorna posizione. Server può scegliere kind dedicato o riusare `agent_action` |
| `storage_changed` | `structure_id,item,qty` | qty **assoluta**, mai delta. qty=0 → cancella la entry |
| `world_event_started` | `event: {id,type,x,y,radius,started_tick,ends_tick}` | aggiunge a `events` |
| `world_event_ended` | `event_id,reason?` | rimuove |
| `dialogue_received` | `from_id,from_name,content` | aggiunto al ring episodico client + ring dedup |
| `gift_received` | `from_id,from_name,item,qty` | ring episodico |
| `loss` | `deceased_id,deceased_name,relation,mood_drop` | ring episodico |
| `user_message` | `content` | ring episodico |
| `relation_update` | `observer_id,target_id,affinity` | client aggiorna `relations` se è il proprio osservatore o `relations_inbound` se è il proprio target |

### Server → Client: result

```json
{"type":"result","tick_ack":N,"action":"...","ok":true|false,"reason":"...","...":...}
```

I campi extra dipendono dall'azione e replicano i return dict di `actions.py` server-side (es. `to`, `from`, `attempted`, `target_name`, `item_type`, `qty`, `materials_used`, `pregnancy_id`, `due_tick`, `cleared_resource`, ecc.).

### Server → Client: heartbeat

```json
{"type":"ping","ts":1714579200.123}
```

Server invia ogni 5s. Client risponde `pong` entro 1s. 3 ping mancati lato server → server fa `wait` per quell'agente fino al reconnect (NON kicka).

### Client → Server: action

```json
{
  "type": "action",
  "tick_ack": 12346,
  "action": "move|wait|wander|gather|eat|craft|build|talk|give|deposit|withdraw|propose|note",
  "direction": "north|south|east|west",
  "target_id": 2,
  "content": "...",
  "item": "...",
  "qty": 1,
  "recipe": "axe|pickaxe|bucket",
  "structure": "hut|storage|shrine",
  "thought": "...",
  "decided_via": "reflex|social|policy|llm|auto_cooldown"
}
```

Solo i campi pertinenti per ogni `action`. `decided_via` è solo telemetria (server non lo usa).

| `action` | Required |
|---|---|
| `move` | `direction ∈ {north,south,east,west}` |
| `wait`, `wander`, `gather` | — |
| `note` | `content` (≤500) |
| `talk` | `target_id`, `content` (3-280 char) |
| `eat` | `item` |
| `craft` | `recipe ∈ {axe,pickaxe,bucket}` |
| `build` | `structure ∈ {hut,storage,shrine}` |
| `give` | `target_id`, `item`, `qty ≥ 1` |
| `deposit`, `withdraw` | `item`, `qty ≥ 1` |
| `propose` | `target_id` |

`thought` ≤ 240 char.

### Client → Server: pong + request_snapshot

```json
{"type":"pong","ts":1714579200.123}
{"type":"request_snapshot"}
```

`request_snapshot` triggers lato client:
1. Tick non monotono crescente (`perception.tick < last_known_tick`).
2. Gap > 60 tick tra perception consecutive.
3. Subito dopo apertura (o riapertura) della WS.

Il server risponde con un nuovo `snapshot` completo.

## Behavior coverage (differenze brain remote vs server)

Il client porta 1:1: reflex priorities, social_navigate, dialogue filters (poetic/italian/anti-noun/anti-trunc/dedup 3-gram), policy (encode/decode/extract_features), prompts (SYSTEM_PROMPT, DIALOGUE_SYSTEM, build_user_prompt, build_dialogue_user_prompt).

NON port (out of scope V1):
- `recall_episodic` DB-backed → ring buffer in-memory (last 30, 12 in prompt).
- `semantic_recall` (sqlite-vec).
- `maybe_reflect` / `maybe_dream` → server li gestisce per gli agenti remoti.
- `wlog_event`, `index_text`, `Observation`/`Dialogue` insert lato client.
- Affinity update lato client. Server applica affinity all'esecuzione delle action e push `relation_update` events.

`LLM_DECIDE_INTERVAL` lato client default = **120 tick** (server usa 300 per via dei vincoli Pi 5).
