# SafeStride App (Next.js + Tailwind)

This is the front-end for SafeStride, built with Next.js App Router and TailwindCSS.

## Quick Start

- Node.js 18+
- From repository root:

```bash
cd app
npm i
npm run dev
```

Then open http://localhost:3000

## Environment

Create `app/.env.local` with:

```
API_BASE=http://localhost:5001
NEXT_PUBLIC_API_BASE=http://localhost:5001
```

- Use the Auth (dev) page to set `SERVICE_TOKEN` (httpOnly cookie). You can also store in localStorage for local-only workflows.
- All client calls go through `/api/proxy` which injects the cookie token.

## Pages

- `/auth` — Set/clear SERVICE_TOKEN for dev.
- `/athlete` — KPI cards, trends (Peak Fz_%BW, Impulse) and a simple live demo calling `/predict` (mocked).
- `/coach` — Leaderboard table from `out_grid_leaderboard_AB01.csv` and an overlay chart (true vs pred) using a sample eval.
- `/admin` — Models registry browser (scans `results/` and `out_grid/` for experiments, surfaces metrics and feature importances if present).
- `/mock` — Loads `out_grid_leaderboard_AB01.csv` into a sortable table.

## Design System

- Primary: #00799C, Blue Light: #4EB5C8
- Neutral Gray bg, Dark text #1C1C1C
- Fonts: Expletus Sans (titles) + Inter (UI)
- 12-col grid, 8px spacing, 8px/12px radii, soft shadows

## Screenshots

Add a screenshot at `docs/app/screenshot.png` showing the dashboard. You can capture after launching `npm run dev`.

## Notes

- File-backed mocks read from the monorepo root (e.g., `results/`, `out_grid/`, `out_grid_leaderboard_AB01.csv`).
- Replace mocks with real endpoints via `API_BASE` and implement `/predict` server route on your API.
