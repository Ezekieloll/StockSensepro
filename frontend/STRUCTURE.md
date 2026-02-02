# Frontend Structure and Conventions

This document defines a clean, uniform structure for the Next.js App Router frontend and gives simple rules for organizing files.

## Top-Level Layout

- `app/`: Route files for the App Router. Each route folder contains `page.tsx` and optional route-local components.
- `components/`: Reusable UI and feature components shared across routes.
- `lib/`: Client-side utilities and helpers (e.g., HTTP client, formatters).
- `public/`: Static assets (images, icons). Served at `/`.
- `globals.css`: Global styles (Tailwind).
- `tsconfig.json`: Path aliases (e.g., `@/*`). Use these for imports.

## Route Organization (`app/`)

- Folders are lowercase and descriptive: `admin/`, `analyst/`, `auth/`, `dashboard/`, `forecasts/`, `manager/`.
- Each route defines one entry: `page.tsx`.
- Route-specific components live under a `components/` subfolder within the route to keep files tidy. Example:
  - `app/analyst/components/AnalysisView.tsx`
  - `app/analyst/components/ProductAccuracyTable.tsx`
  - `app/analyst/components/AccuracySummaryCard.tsx`
  - `app/analyst/components/GNN3DVisualizer.tsx`
- Keep server components as default; add `'use client'` only where needed.

## Shared Components (`components/`)

- Use PascalCase file names for React components.
- Split into two categories for clarity:
  - `components/ui/`: Visual primitives with minimal domain logic.
    - Examples: `Button.tsx`, `Badge.tsx`, `Card.tsx`, `Input.tsx`, `Select.tsx`, `Table.tsx`, `Icons.tsx`.
  - `components/features/`: Domain-specific composites with business logic.
    - Examples: `InsightAssistant.tsx`.
- Prefer named exports for multi-part components (e.g., `CardHeader`, `CardContent`) to avoid deep imports.
- Optional: Add a `components/index.ts` barrel to streamline imports, but direct imports like `@/components/ui/Button` are fine.

## Utilities (`lib/`)

- Place shared helpers like API clients in `lib/`.
- Use path alias imports: `import api from '@/lib/api'`.
- Keep utilities framework-agnostic when possible.

## Static Assets (`public/`)

- Put images and icons here; reference via `/file.svg` or `next/image`.
- Keep app metadata assets (e.g., `favicon.ico`) in `app/` per Next.js guidelines.

## Import Rules

- Use the `@/*` path alias (set in `tsconfig.json`) for all cross-folder imports.
  - Good: `import Button from '@/components/ui/Button'`.
  - Avoid deep relative paths like `../../components/...`.

## Client vs Server Components

- Default to server components for pages and shared UI when possible.
- Add `'use client'` only for components using state, effects, or browser-only APIs.

## Naming & Style

- Components: PascalCase (`MyWidget.tsx`).
- Routes: lowercase directories (`analyst`, `forecasts`).
- Files exporting multiple symbols can use PascalCase or domain name-based (`Icons.tsx`).
- Keep files small and cohesive; extract UI primitives into `components/ui`.

## Recommended Near-Term Cleanup

- Move feature-level components into `components/features/` (e.g., `InsightAssistant.tsx`).
- Create `components/ui/` and relocate UI primitives (`Button`, `Badge`, `Card`, `Input`, `Select`, `Table`, `Icons`).
- Under each route, add `components/` subfolder and relocate route-local components currently next to `page.tsx`.
- Update imports to use the `@/*` aliases consistently after moves.

## Example Structure (target)

```
frontend/
  app/
    analyst/
      components/
        AnalysisView.tsx
        ProductAccuracyTable.tsx
        AccuracySummaryCard.tsx
        GNN3DVisualizer.tsx
      page.tsx
    admin/
      page.tsx
    auth/
      login/
        page.tsx
      signup/
        page.tsx
    dashboard/
      page.tsx
    forecasts/
      page.tsx
    manager/
      page.tsx
    globals.css
    layout.tsx
  components/
    ui/
      Button.tsx
      Badge.tsx
      Card.tsx
      Input.tsx
      Select.tsx
      Table.tsx
      Icons.tsx
    features/
      InsightAssistant.tsx
  lib/
    api.ts
  public/
    ...
  tsconfig.json
  eslint.config.mjs
  next.config.ts
```

If you’d like, I can apply these moves and update imports in one pass.