## group-elephant 

### Split of work (4 people)
- **Workstream A – Data**: `src/group_elephant/workstreams/data/`
- **Workstream B – Modeling**: `src/group_elephant/workstreams/modeling/`
- **Workstream C – Backend/API**: `src/group_elephant/workstreams/backend/`
- **Workstream D – Frontend/Visualization**: `web/` (and optional Python helpers under `src/group_elephant/workstreams/frontend/`)


### Python setup (uv)
Requires Python 3.11+ and `uv`.

- Create & sync env:

```bash
uv sync
```

- Run tests:

```bash
uv run pytest
```

### JS setup (pnpm)
Requires `pnpm`.

```bash
pnpm -w install
```



