# GUI-only Business Process Graph for Domain-aware LLM Agents

## Overview

This repository contains a research-oriented prototype developed as part of a
master’s thesis.

The goal is to make LLM-based agents **domain-aware** by providing them with an
explicit representation of business logic inferred **only from GUI observations**.

No backend access.
No APIs.
No business documentation.

Only:
- screenshots,
- OCR,
- clickstreams.

---

## Motivation

Modern LLM agents:
- can interpret text and images,
- but lack stable understanding of business constraints.

As a result:
- syntactically correct actions are often semantically invalid,
- RPA pipelines are brittle and UI-dependent,
- failures are hard to explain or recover from.

This project addresses the problem by introducing a **Business Process Graph (BPG)**
as a runtime knowledge base.

---

## Core Idea

> Explicit domain knowledge + adaptive LLM reasoning  
> = robust, explainable automation

BPG captures:
- entities,
- actions,
- workflows,
- constraints,
- roles,

and exposes them to LLM agents at runtime.

---

## System Scope

### What the system CAN access
- GUI screenshots
- OCR text
- User clickstreams / session traces

### What the system CANNOT access
- Backend APIs
- Databases
- DOM / HTML
- Source code
- Business documentation

---

## High-level Pipeline

Screenshots + Clickstreams
→ GUI element detection
→ Multimodal representations
→ Cross-view entity linking
→ Action & pattern induction
→ Business Process Graph (BPG)
→ Runtime LLM context

---

## Business Process Graph (BPG)

BPG is a graph-based knowledge representation with:

- Nodes:
    - EntityType
    - EntityInstance
    - GUIManifestation
    - Action
    - PatternNode
    - Rule
- Edges:
    - cross_view
    - functional
    - temporal
    - conditional
    - compositional
    - role

Formal schema: see `BPG_SCHEMA.md`.

---

## Architecture & Documentation

- `PROJECT_CONTEXT.md` — problem statement and research framing
- `ARCHITECTURE.md` — system modules and data flow
- `BPG_SCHEMA.md` — formal graph schema
- `TECH_STACK.md` — implementation technologies
- `diagrams/bpg_rkb/` — architectural and schema diagrams

---

## Design Principles

- GUI-only evidence
- Explicit uncertainty (confidence scores)
- Provenance for all inferred knowledge
- Modular, research-friendly architecture
- Explainability over performance

This is **not** a production RPA framework.

---

## Research Focus

The project evaluates:
- feasibility of GUI-only business logic extraction,
- robustness of LLM planning with BPG context,
- explainability and recovery behavior,
- trade-offs between heuristic, ML, and LLM-assisted inference.

---

## Status

Current focus:
- minimal end-to-end PoC,
- clear abstractions,
- experimental evaluation.

The codebase prioritizes clarity and extensibility over optimization.
