# Business Process Graph (BPG) Schema

## Purpose

Business Process Graph (BPG) is an explicit, runtime knowledge representation of
business logic inferred **only from GUI observations and user behavior**.

BPG is NOT:
- a static documentation artifact
- a BPMN diagram
- a backend-derived schema

BPG IS:
- a runtime knowledge base for LLM agents
- a constraint system for action planning
- an explainability surface

---

## Core Principles

- GUI-only evidence (screenshots, OCR, clickstreams)
- Explicit uncertainty (confidence scores)
- Provenance for every inferred fact
- Separation between *what is observed* and *what is inferred*

---

## Node Types

### 1. EntityType
Represents a business-level concept inferred from GUI.

Examples:
- Product
- Order
- User
- Invoice
- Cart

Attributes:
- `id`
- `name`
- `description`
- `confidence`
- `provenance`

---

### 2. EntityInstance
Concrete instance of an EntityType observed in GUI.

Examples:
- Product #123
- Order #A-456

Attributes:
- `id`
- `entity_type_id`
- `attributes` (key-value, inferred)
- `confidence`
- `provenance`

---

### 3. GUIManifestation
A concrete visual representation of an entity in a specific GUI view.

Examples:
- Product card in catalog
- Product row in table
- Product summary in checkout

Attributes:
- `id`
- `entity_instance_id`
- `screenshot_id`
- `bounding_box`
- `visual_embedding`
- `text_embedding`
- `layout_features`

---

### 4. Action
An action affordance inferred from GUI elements.

Examples:
- Click "Add to cart"
- Submit form
- Select dropdown value

Attributes:
- `id`
- `action_type`
- `trigger_element`
- `confidence`
- `provenance`

---

### 5. PatternNode
Represents a higher-level workflow or reusable interaction pattern.

Examples:
- Checkout flow
- Authentication flow
- Item creation workflow

Attributes:
- `id`
- `name`
- `steps` (ordered references)
- `confidence`

---

### 6. Rule
Explicit constraint or condition inferred from GUI behavior.

Examples:
- User must be authenticated to see Cart
- Order can be created only if Cart is non-empty

Attributes:
- `id`
- `rule_type`
- `condition`
- `scope`
- `confidence`
- `provenance`

---

## Edge Types

### cross_view
Links different GUIManifestations of the same EntityInstance.

Example:
```

GUIManifestation(A) --cross_view--> GUIManifestation(B)

```

---

### compositional
Represents containment or structural hierarchy.

Examples:
- Order contains OrderItem
- Table contains Rows

---

### functional
Represents action → result relation.

Example:
```

Action(AddToCart) --functional--> EntityInstance(Order)

```

---

### temporal
Represents frequent or causal ordering inferred from clickstreams.

Example:
```

View(Product) --temporal--> View(Cart)

```

---

### conditional
Represents preconditions and postconditions.

Example:
```

Rule(AuthRequired) --conditional--> Action(Checkout)

```

---

### role
Assigns semantic role to GUI elements.

Examples:
- filter
- selector
- primary_action
- info_display

---

## Provenance Model

Every node and edge must store:
- evidence sources (screenshots, sessions)
- inference method (heuristic / ML / LLM)
- confidence score ∈ [0, 1]

No edge exists without provenance.

---

## Example (Simplified)

EntityType(Product)
↓
EntityInstance(Product#123)
↓ cross_view
GUIManifestation(CatalogCard)
GUIManifestation(CheckoutRow)

Action(AddToCart)
↓ functional
EntityInstance(Order#456)

Rule(AuthRequired)
↓ conditional
Action(Checkout)

---

## Runtime Usage

BPG is queried at runtime to:
- validate LLM-generated action plans
- enrich prompts with domain constraints
- explain *why* an action is valid or invalid
