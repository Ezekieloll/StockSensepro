# Adversarial Scenario Generation for Inventory Stress Testing: A Multi-Layer Framework

---

**Abstract** — This document presents the design, mathematical formulation, and implementation of the adversarial scenario generation engine within the StockSense inventory management system. The engine is a multi-layer stress-testing framework that evaluates supply chain resilience by systematically perturbing forecasting model inputs and computing downstream inventory risk under simulated disruption conditions. The architecture comprises four progressively sophisticated layers: (i) a deterministic perturbation engine that operates directly on the trained TFT–GNN forecasting model's input tensor, (ii) a rule-based scenario library encoding eight canonical retail disruption archetypes with calibrated demand multipliers and duration profiles, (iii) an LLM-augmented dynamic scenario generator that uses a local Qwen 2.5 language model to synthesize context-aware stress scenarios from live database analytics, and (iv) a natural-language interactive analyst that converts free-text "what-if" queries into quantified impact assessments. We formalize the risk quantification metrics—stockout detection, severity estimation, days-of-cover analysis, and composite risk scoring—and describe the graph-propagated impact analysis that leverages the TFT–GNN product adjacency matrix to model cross-SKU demand contagion. The system operates across 240 SKUs, 3 stores, and 8+ scenario types, persisting results to a PostgreSQL database and exposing them via a RESTful API for real-time decision support.

---

## 1. Introduction

### 1.1 Motivation

Demand forecasting models, regardless of accuracy, produce point estimates conditioned on an assumed distribution of future inputs. In operational settings, the primary risk is not forecast inaccuracy under normal conditions, but forecast failure under *distributional shift*—sudden, non-stationary demand shocks caused by external events (holidays, weather, supply disruptions, competitive dynamics). Traditional safety-stock heuristics address this with static buffers, but are poorly calibrated to the heterogeneous risk profiles of diverse SKU catalogs.

Adversarial scenario testing provides a principled framework for evaluating inventory resilience by answering the question: *given a plausible disruption, which SKU–store pairs will experience stockout, and what is the expected severity?*

### 1.2 Design Philosophy

The adversarial engine is designed around three principles:

1. **Determinism over stochasticity.** Monte Carlo simulation produces noisy risk estimates that require thousands of draws for convergence. The adversarial engine instead constructs a *finite set of worst-case scenarios*, each grounded in an economically meaningful disruption archetype. This yields interpretable, reproducible risk assessments with a single forward pass per scenario.

2. **Layered sophistication.** The four-layer architecture allows the system to operate at different fidelity levels depending on available infrastructure. Layer 1 (tensor perturbation) requires only the trained PyTorch model. Layers 2–3 add scenario structure and AI-driven contextualization. Layer 4 provides conversational access for non-technical users. Each layer degrades gracefully when upstream dependencies are unavailable.

3. **Graph-propagated impact.** By reusing the product adjacency matrix $A \in \mathbb{R}^{240 \times 240}$ from the TFT–GNN forecasting model, the adversarial engine can propagate demand shocks across related SKUs—capturing substitution effects, complementary demand cascades, and category-level disruptions that isolated per-SKU analysis would miss.

### 1.3 Document Organization

Section 2 formalizes the problem setting. Section 3 describes the four-layer architecture in detail. Section 4 derives the risk quantification metrics. Section 5 presents the inter-store rebalancing algorithm. Section 6 documents the database persistence layer and API integration. Section 7 discusses system-level design decisions and trade-offs.

---

## 2. Problem Formulation

### 2.1 Notation

| Symbol | Definition |
|--------|-----------|
| $\mathbf{x} \in \mathbb{R}^{B \times T \times F}$ | Input tensor: batch of $B$ sequences, each $T = 30$ time steps with $F = 9$ features |
| $f_\theta(\cdot)$ | Trained TFT–GNN forecasting model with parameters $\theta$ |
| $\hat{y} = f_\theta(\mathbf{x})$ | Baseline demand forecast (scalar per sample) |
| $\hat{y}^{(s)} = f_\theta(\mathbf{x}^{(s)})$ | Adversarial forecast under scenario $s$ |
| $I_{k,j}$ | Current on-hand inventory for SKU $k$ at store $j$ |
| $A \in \mathbb{R}^{N \times N}$ | Weighted product adjacency matrix ($N = 240$) |
| $\mathcal{S} = \{s_1, \dots, s_M\}$ | Set of $M$ adversarial scenarios |
| $\mu_s$ | Demand multiplier for scenario $s$ |
| $\Delta_s$ | Duration (days) of scenario $s$ |
| $p_s$ | Prior probability of scenario $s$ occurring |

### 2.2 Objective

For each SKU–store pair $(k, j)$ and each scenario $s \in \mathcal{S}$, the adversarial engine computes:

1. **Worst-case demand** $\hat{y}^{(s)}_{k,j}$: the expected daily demand under scenario $s$.
2. **Stockout indicator** $\mathbb{1}[\hat{y}^{(s)}_{k,j} > I_{k,j}]$: whether current inventory is insufficient.
3. **Risk score** $r_{k,j}^{(s)} \in [0, 1]$: a normalized severity measure.
4. **Strategic recommendations**: actionable mitigation strategies indexed by scenario type and priority.

The system then aggregates across scenarios to identify the most vulnerable SKU–store pairs and the most critical scenario by combined risk:

$$
s^* = \arg\max_{s \in \mathcal{S}} \; p_s \cdot \frac{1}{|\mathcal{K}_s|} \sum_{(k,j) \in \mathcal{K}_s} \mathbb{1}[\text{stockout}_{k,j}^{(s)}]
$$

where $\mathcal{K}_s$ is the set of SKU–store pairs affected by scenario $s$.

---

## 3. Multi-Layer Architecture

### 3.1 Layer 1: Deterministic Perturbation Engine (`ScenarioSimulator`)

The lowest layer operates directly on the forecasting model's input tensor $\mathbf{x} \in \mathbb{R}^{B \times T \times F}$, applying controlled perturbations to specific feature channels and measuring the model's response. This layer requires no external data or LLM infrastructure—only the trained PyTorch model in evaluation mode.

#### 3.1.1 Perturbation Operators

Five deterministic perturbation operators are defined, each targeting a specific feature index (per the feature schema in Table 1 of the forecasting model documentation):

**Operator 1: Demand Spike.** Multiplicative scaling of the historical demand channel (feature index 0):

$$
\mathbf{x}^{(\text{spike})}_{b, t, 0} = \alpha \cdot \mathbf{x}_{b, t, 0}, \quad \forall\, b, t
$$

where $\alpha > 1$ is the spike factor (default $\alpha = 1.5$). This simulates a scenario where recent demand history indicates an upward shock, and the model must extrapolate whether the trend continues.

**Operator 2: Demand Drop.** Symmetric to the demand spike:

$$
\mathbf{x}^{(\text{drop})}_{b, t, 0} = \beta \cdot \mathbf{x}_{b, t, 0}, \quad \forall\, b, t
$$

where $0 < \beta < 1$ is the drop factor (default $\beta = 0.5$). This tests the model's response to demand contraction events (e.g., economic downturn, competitor entry).

**Operator 3: Weather Shock.** Replacement of the weather feature (index 3) with an extreme condition code:

$$
\mathbf{x}^{(\text{weather})}_{b, t, 3} = w_{\text{target}}, \quad \forall\, b, t
$$

where $w_{\text{target}} \in \{0, 1, 2, 3\}$ follows the encoding (Cloudy=0, Rainy=1, Storm=2, Sunny=3). Setting $w_{\text{target}} = 2$ (Storm) across the entire 30-day window simulates a sustained severe weather event.

**Operator 4: Holiday Shock.** Activation of the holiday indicator (index 2) across all time steps:

$$
\mathbf{x}^{(\text{holiday})}_{b, t, 2} = 1.0, \quad \forall\, b, t
$$

This simulates an extended holiday period where every day in the forecast window is flagged as a public holiday.

**Operator 5: Worst-Case Envelope.** The maximum forecast across all perturbation operators:

$$
\hat{y}^{(\text{worst})} = \max\!\left(\hat{y}^{(\text{baseline})},\; \hat{y}^{(\text{spike}, 1.5)},\; \hat{y}^{(\text{spike}, 2.0)},\; \hat{y}^{(\text{weather})},\; \hat{y}^{(\text{holiday})}\right)
$$

This computes the upper demand envelope by selecting the scenario that produces the highest predicted demand for each sample. The worst-case envelope provides a conservative inventory buffer recommendation that covers all tested perturbations simultaneously.

#### 3.1.2 Inference Protocol

All perturbations follow the same stateless inference protocol:

1. Clone the input tensor to prevent mutation: $\mathbf{x}' = \text{clone}(\mathbf{x})$.
2. Apply the perturbation operator to the cloned tensor.
3. Run forward inference with gradients disabled: $\hat{y}^{(s)} = f_\theta(\mathbf{x}')$ under `torch.no_grad()`.
4. Return predictions on CPU for downstream risk evaluation.

The model is held in `eval()` mode throughout. No training, backpropagation, or parameter update occurs during adversarial evaluation.

---

### 3.2 Layer 2: Rule-Based Scenario Library (`AIScenarioGenerator`)

The second layer encapsulates domain knowledge in a library of eight canonical retail disruption scenarios. Each scenario is parameterized by a `Scenario` dataclass:

```
@dataclass
class Scenario:
    id: str                          # Unique identifier (snake_case)
    name: str                        # Human-readable name
    description: str                 # 2–3 sentence narrative
    demand_multiplier: float         # Expected demand scale factor (μ_s)
    duration_days: int               # Scenario duration (Δ_s)
    affected_categories: List[str]   # Targeted product categories
    probability: float               # Prior likelihood p_s ∈ [0, 1]
    strategies: List[str]            # Mitigation recommendations
    priority_level: str              # {critical, high, medium, low}
```

#### 3.2.1 Scenario Catalog

**Table 1: Canonical Disruption Scenarios**

| ID | Name | $\mu_s$ | $\Delta_s$ (days) | $p_s$ | Priority | Affected Categories |
|----|------|---------|--------------------|--------|----------|---------------------|
| `holiday_rush` | Holiday Shopping Rush | 4.0× | 14 | 0.90 | Critical | Fresh Produce, Bakery, Beverages, Dairy |
| `supply_disruption` | Supply Chain Disruption | 1.0× | 7 | 0.60 | High | All |
| `viral_trend` | Viral Social Media Trend | 15.0× | 5 | 0.30 | Medium | Random (unpredictable) |
| `weather_emergency` | Severe Weather Event | 2.5× | 3 | 0.40 | High | Fresh Produce, Beverages, Canned Goods |
| `competitor_closure` | Competitor Store Closure | 1.4× | 90 | 0.20 | Medium | All |
| `product_recall` | Competitor Product Recall | 6.5× | 21 | 0.25 | High | Category-specific |
| `economic_boom` | Local Economic Boom | 1.25× | 180 | 0.15 | Low | All |
| `promo_event` | Major Promotional Campaign | 5.0× | 7 | 0.80 | Medium | Promotional categories |

The demand multiplier $\mu_s$ is applied to the historical average daily demand $\bar{d}_{k,j}$ to compute worst-case demand:

$$
\hat{y}^{(s)}_{k,j} = \mu_s \cdot \bar{d}_{k,j}
$$

Note that `supply_disruption` uses $\mu_s = 1.0$: demand remains constant, but the disruption lies in the *inability to restock*, making even normal demand levels hazardous when inventory buffers are thin.

#### 3.2.2 Vulnerability Analysis

The `analyze_vulnerability()` method evaluates a single SKU across all applicable scenarios by computing:

1. **Category matching.** Each scenario specifies affected product categories. Scenarios with `affected_categories = ["All"]` or `["Random"]` apply universally; others apply only when the SKU's category matches.

2. **Days-of-cover computation.**

$$
\text{DoC}^{(s)}_{k,j} = \frac{I_{k,j}}{\hat{y}^{(s)}_{k,j}}
$$

3. **Stockout risk detection.** A stockout is flagged when days-of-cover is less than the scenario's duration:

$$
\text{stockout}^{(s)}_{k,j} = \mathbb{1}\!\left[\text{DoC}^{(s)}_{k,j} < \Delta_s\right]
$$

4. **Overall risk scoring.** The weighted aggregate risk for a SKU across all scenarios:

$$
R_{k,j} = \frac{\sum_{s \in \mathcal{S}} p_s \cdot \mathbb{1}[\text{stockout}^{(s)}_{k,j}]}{\sum_{s \in \mathcal{S}} p_s}
$$

This probability-weighted ratio gives $R_{k,j} \in [0, 1]$, where $R = 1$ means the SKU would experience stockout under every scenario, weighted by each scenario's likelihood.

---

### 3.3 Layer 3: LLM-Augmented Dynamic Scenario Generation (`dynamic_ai_scenarios`)

The third layer augments the static scenario library with dynamically generated scenarios conditioned on live database analysis. It uses a locally hosted Qwen 2.5 (7B parameter) language model via Ollama to synthesize scenarios that reflect the *current* state of the business.

#### 3.3.1 Database Pattern Analysis

Before querying the LLM, the system executes a structured database analysis pipeline:

**Step 1: Demand Volatility Profiling.** For each SKU–store pair, compute the coefficient of variation (CV) and spike ratio over the trailing 30-day window:

$$
\text{CV}_{k,j} = \frac{\sigma_{k,j}}{\bar{d}_{k,j}}, \qquad \text{SpikeRatio}_{k,j} = \frac{\max_t\, d_{k,j,t}}{\bar{d}_{k,j}}
$$

SKU–store pairs with $\text{CV} > 0.5$ or $\text{SpikeRatio} > 3.0$ are flagged as high-volatility and surfaced to the LLM as context. These thresholds are calibrated to identify the tail of the demand distribution—approximately the top 10–15% of SKUs by variability.

**Step 2: Inventory Risk Screening.** For each inventory record, compute days-of-cover against average demand:

$$
\text{DoC}_{k,j} = \frac{I_{k,j}}{\bar{d}_{k,j}}
$$

Items with $\text{DoC} < 7$ days are flagged as low-stock and included in the LLM prompt. The 7-day threshold corresponds to a standard weekly replenishment cycle in grocery retail.

**Step 3: Category Mix Analysis.** Aggregate sales volume by product category over the trailing 30 days, sorted by total units sold. The top 5 categories are provided to the LLM to contextualize where demand concentration lies.

#### 3.3.2 LLM Prompt Engineering

The system constructs a two-part prompt:

- **System prompt**: Defines the role (supply chain analyst), output format (strict JSON array of 5 scenarios), schema contract (matching the `Scenario` dataclass fields), and severity distribution constraint (1 critical, 2 high, 1 medium, 1 low).

- **User prompt**: Injects the structured database analysis—high-volatility products, low-stock items, top categories—as raw JSON. The prompt instructs the LLM to:
  1. Use *actual* spike ratios from the data as calibration for demand multipliers.
  2. Address observed low-stock vulnerabilities.
  3. Reference real SKU IDs from the database (with `SKU_` prefix normalization).
  4. Provide actionable strategies contextualized to the specific business.

#### 3.3.3 Response Extraction and Normalization

LLM responses are inherently variable in format. The extraction pipeline employs a multi-strategy parser:

1. Attempt direct JSON parse of the full response.
2. Extract fenced code blocks (` ```json ... ``` `).
3. Heuristic extraction of the first JSON array (`[...]`) or object (`{...}`) substring.
4. For parsed objects, check common wrapper keys: `scenarios`, `data`, `result`, `items`.
5. Accept single-scenario objects by wrapping in a list.

Successfully extracted scenarios undergo normalization:

- **Type coercion**: `demand_multiplier` → `float`, `duration_days` → `int`, `probability` → `float`, with safe defaults on failure.
- **SKU reference normalization**: A regex pattern converts shorthand references like `FRPR002` to the canonical `SKU_FRPR002` format.
- **Priority validation**: Clamped to `{critical, high, medium, low}`, defaulting to `medium`.
- **Strategy normalization**: Accepts both JSON arrays and pipe-separated strings.

#### 3.3.4 Graceful Degradation

If the LLM is unavailable (Ollama not running), times out, or produces unparseable output, the system falls back to the Layer 2 rule-based scenario library. This ensures the adversarial engine is always operational regardless of LLM infrastructure status.

---

### 3.4 Layer 4: Interactive Natural-Language Scenario Analysis (`interactive_scenario_ai`)

The fourth layer provides a conversational interface for free-text "what-if" scenario exploration. Unlike Layers 2–3, which generate structured scenarios for batch processing, Layer 4 accepts unstructured natural language inputs and produces narrative-form impact analyses.

#### 3.4.1 Input Examples

- *"Tomorrow there will be a lockdown due to health emergency"*
- *"Weather forecast shows heavy snowstorm for next 3 days"*
- *"News: 15% tax increase on all grocery items from next week"*
- *"Competitor store closing next week in S1 area"*

#### 3.4.2 Analysis Pipeline

1. **Context assembly.** For each store, the system queries current inventory levels, average daily demand per SKU (trailing 30 days), and category-level sales rankings. This context is serialized as JSON.

2. **LLM analysis.** The assembled context and user query are passed to Qwen 2.5 with a system prompt that requests:
   - Summary of the scenario's business impact.
   - Store-specific impact assessment (which of S1, S2, S3 are affected).
   - Estimated demand multiplier and duration.
   - SKU-level risk identification using actual SKU IDs from the database.
   - Store-specific tactical recommendations.

3. **Structured impact quantification.** The LLM's qualitative analysis is translated into quantitative impact metrics:

$$
d^{(s)}_{k,j} = \mu_s \cdot \bar{d}_{k,j}, \qquad D^{(s)}_{\text{total}} = d^{(s)}_{k,j} \cdot \Delta_s
$$

$$
\text{Shortage}_{k,j} = \max\!\left(0,\; D^{(s)}_{\text{total}} - I_{k,j}\right)
$$

$$
\text{DaysUntilStockout}_{k,j} = \frac{I_{k,j}}{d^{(s)}_{k,j}}
$$

Each SKU is assigned an action urgency level:

| Days Until Stockout | Action Level |
|---------------------|-------------|
| $< 2$ | **URGENT** |
| $2 \leq \cdot < 5$ | MODERATE |
| $\geq 5$ | MONITOR |

---

## 4. Risk Quantification Framework (`InventoryRiskEvaluator`)

The `InventoryRiskEvaluator` provides a deterministic, tensor-compatible risk computation that serves as the quantitative backbone for all four layers.

### 4.1 Inputs

| Parameter | Symbol | Type |
|-----------|--------|------|
| Baseline daily demand | $\hat{y}_{\text{base}}$ | `Tensor` or `float` |
| Worst-case daily demand | $\hat{y}_{\text{worst}}$ | `Tensor` or `float` |
| Current inventory level | $I$ | `Tensor` or `float` |

All inputs are cast to `torch.float32` for uniform handling, enabling both scalar and batched evaluation.

### 4.2 Computed Metrics

**Stockout Indicator.** Binary detection of whether worst-case demand exceeds current inventory:

$$
\text{stockout} = \mathbb{1}\!\left[\hat{y}_{\text{worst}} > I\right]
$$

**Severity.** The expected inventory shortfall (in demand units), clamped to non-negative:

$$
\text{severity} = \max\!\left(0, \; \hat{y}_{\text{worst}} - I\right)
$$

This represents the number of demand units that cannot be fulfilled under the worst-case scenario.

**Days of Cover.** The number of days current inventory can sustain baseline demand:

$$
\text{DoC} = \frac{I}{\hat{y}_{\text{base}} + \epsilon}
$$

where $\epsilon = 10^{-6}$ prevents division by zero for zero-demand SKUs. Note that DoC is computed against *baseline* demand (not worst-case), providing a normalized measure of buffer adequacy.

**Risk Score.** A normalized composite metric in $[0, 1]$:

$$
r = \text{clamp}\!\left(\frac{\text{severity}}{\hat{y}_{\text{worst}} + \epsilon}, \; 0, \; 1\right)
$$

The risk score is the fraction of worst-case demand that cannot be satisfied. $r = 0$ indicates full coverage; $r = 1$ indicates zero inventory against non-zero demand.

---

## 5. Inter-Store Inventory Rebalancing (`InventoryRebalancer`)

When adversarial analysis reveals stockout risk at specific stores, the rebalancing module computes an optimal inter-store redistribution plan that mitigates risk without requiring external procurement.

### 5.1 Classification Phase

For a given SKU, each store is classified as *surplus* or *deficit* based on the gap between current inventory and worst-case demand:

$$
\text{surplus}_j = I_j - \hat{y}^{(\text{worst})}_j \quad \text{if } I_j > \hat{y}^{(\text{worst})}_j
$$

$$
\text{deficit}_j = \hat{y}^{(\text{worst})}_j - I_j \quad \text{if } I_j < \hat{y}^{(\text{worst})}_j
$$

If no surplus or no deficit stores exist, no transfers are required.

### 5.2 Greedy Transfer Algorithm

The algorithm employs a greedy two-pointer strategy:

1. Sort deficit stores by deficit magnitude (descending—largest deficits first).
2. Sort surplus stores by surplus magnitude (descending—largest surpluses first).
3. For each deficit store, greedily transfer units from the current surplus store:

$$
\text{transfer}_{i \to j} = \min\!\left(\text{remaining\_deficit}_j, \; \text{remaining\_surplus}_i\right)
$$

4. When a surplus store is exhausted, advance to the next surplus store.
5. Terminate when all deficits are satisfied or all surplus is consumed.

**Output.** A list of transfer actions:
```
[{"sku": "SKU_FRPR002", "from": "S1", "to": "S3", "units": 15}, ...]
```

### 5.3 Optimality

The greedy algorithm produces an optimal solution (minimum total transfers) when the objective is to maximize deficit coverage. This is a special case of the transportation problem where all supply and demand nodes are on a single commodity, and the cost function is uniform across all origin–destination pairs.

---

## 6. Category-Scoped Scenario Execution (`populate_db_ai`)

The full adversarial testing pipeline integrates Layers 2–3 with the risk evaluation framework and persists results to a PostgreSQL database.

### 6.1 Category Matching

Each scenario specifies affected product categories. To bridge the gap between scenario-level category labels (e.g., "Fresh Produce") and database-level category codes, the system implements a fuzzy matching pipeline:

1. **Normalization.** Both scenario labels and database categories are lowercased and stripped of non-alphanumeric characters: `"Fresh Produce"` → `"freshproduce"`.

2. **Alias expansion.** A hand-curated alias map handles common synonyms:
   - `"freshproduce"` → `{"freshproduce", "produce", "fresh", "fruits", "vegetables", "veg"}`
   - `"beverages"` → `{"beverages", "beverage", "drinks", "drink"}`
   - `"dairy"` → `{"dairy", "milk", "eggs"}`

3. **Wildcard categories.** Scenarios with `affected_categories = ["All"]` or `["Random"]` bypass category filtering entirely.

4. **Meta-categories.** The label `"Promotional categories"` maps to the top 3 database categories by sales volume. `"Specific affected category"` maps to the single highest-volume category.

5. **Partial match fallback.** If exact match fails, substring containment is checked bidirectionally.

### 6.2 Execution Flow

For each scenario $s \in \mathcal{S}$:

1. Retrieve all inventory records from the database.
2. Build a demand baseline cache: 30-day trailing average of daily demand per SKU–store pair from the `daily_demand` table.
3. Build a category lookup per SKU–store pair from historical demand records.
4. For each inventory record $(k, j)$ where $\text{category}(k) \in \text{affected}(s)$:
   - Compute $\hat{y}^{(s)}_{k,j} = \mu_s \cdot \bar{d}_{k,j}$.
   - Evaluate risk via `InventoryRiskEvaluator`.
   - Create a database record with scenario metadata, risk metrics, and strategic recommendations.
5. Bulk-insert all records and commit.

### 6.3 Custom Scenario Support

The pipeline accepts optional `custom_scenarios` alongside the built-in library. Custom scenarios follow the same schema and can override existing scenarios by ID. This enables frontline users to define ad-hoc stress tests (e.g., a regional event specific to their market) without modifying source code.

### 6.4 Critical Scenario Identification

After testing all scenarios, the system identifies the most critical scenario by the combined metric:

$$
\text{CombinedRisk}(s) = p_s \times \frac{|\{(k,j) : \text{stockout}^{(s)}_{k,j}\}|}{|\mathcal{K}_s|}
$$

This probability-weighted stockout rate reflects both the likelihood and the breadth of impact.

---

## 7. Database Schema and API Integration

### 7.1 Persistence Layer

Risk assessment results are persisted in the `adversarial_risk` PostgreSQL table:

**Table 2: Database Schema — `adversarial_risk`**

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Auto-increment primary key |
| `sku` | String (indexed) | Product SKU identifier |
| `sku_id` | String (indexed) | Redundant SKU reference for consistency |
| `store_id` | String (indexed) | Store identifier (S1, S2, S3) |
| `scenario_name` | String | Human-readable scenario name |
| `scenario_id` | String (indexed) | Machine identifier (snake_case) |
| `probability` | Float | Scenario prior probability $p_s$ |
| `strategies` | Text | Pipe-separated mitigation strategies |
| `priority_level` | String | Severity tier: critical, high, medium, low |
| `baseline_demand` | Float | 30-day average daily demand $\bar{d}_{k,j}$ |
| `worst_case_demand` | Float | Scenario-adjusted demand $\hat{y}^{(s)}_{k,j}$ |
| `current_inventory` | Integer | On-hand inventory at time of evaluation |
| `severity` | Float | Inventory shortfall (units) |
| `days_of_cover` | Float | Baseline DoC |
| `risk_score` | Float | Normalized risk $r \in [0, 1]$ |
| `stockout` | Boolean | Stockout indicator |
| `created_at` | DateTime | Timestamp of risk assessment |

### 7.2 REST API Endpoints

The adversarial engine is exposed through a FastAPI router mounted at `/api/adversarial`:

**Table 3: API Endpoints**

| Method | Path | Authentication | Description |
|--------|------|----------------|-------------|
| `GET` | `/adversarial/scenarios` | None | List available scenarios. `?use_ai=true` triggers LLM-generated scenarios. |
| `POST` | `/adversarial/run-ai-test` | JWT required | Execute adversarial testing with selected/custom scenarios. |
| `GET` | `/adversarial/` | None | Query persisted risk records with filters: `sku`, `store_id`, `scenario_id`, `high_risk_only`. |
| `POST` | `/adversarial/run-test` | JWT required | Execute the deterministic (Layer 1) adversarial test pipeline. |
| `POST` | `/adversarial/scenario-activity` | JWT required | Log scenario editor activity (create/update/delete) for audit trail. |

All write operations are audit-logged with user ID, IP address, and outcome details.

### 7.3 Request and Response Schemas

**Run AI Test Request:**
```json
{
  "scenario_ids": ["holiday_rush", "weather_emergency"],
  "category_scoped": true,
  "custom_scenarios": [
    {
      "id": "local_flood",
      "name": "Local Flooding Event",
      "description": "Regional flooding disrupts supply routes for 5 days",
      "demand_multiplier": 3.0,
      "duration_days": 5,
      "affected_categories": ["Fresh Produce", "Beverages"],
      "probability": 0.15,
      "strategies": ["Pre-position emergency stock", "Activate backup suppliers"],
      "priority_level": "high"
    }
  ]
}
```

**AI Test Response:**
```json
{
  "status": "success",
  "scope_mode": "strict",
  "scenarios_tested": 3,
  "total_records": 720,
  "results_by_scenario": {
    "holiday_rush": {
      "name": "Holiday Shopping Rush",
      "records_tested": 240,
      "stockout_count": 180,
      "stockout_rate": 0.75,
      "avg_risk_score": 0.68,
      "probability": 0.90,
      "strategies": ["Increase safety stock by 200%...", "..."]
    }
  },
  "most_critical_scenario": { "..." }
}
```

---

## 8. Integration with the TFT–GNN Forecasting Model

### 8.1 Graph-Propagated Demand Shock Analysis

A key differentiator of the adversarial engine is its ability to propagate demand shocks through the product graph. The TFT–GNN forecasting model constructs a weighted adjacency matrix $A$ from co-purchase, category, and correlation signals (detailed in the forecasting model documentation, §4). The adversarial engine reuses this graph for *cross-SKU impact analysis*:

When a scenario targets a specific product category, the demand multiplier is applied directly to affected SKUs. However, the GNN message-passing operation:

$$
\mathbf{H}' = \sigma\!\left(\hat{A}\, \mathbf{H}\, \mathbf{W}\right)
$$

ensures that the *forecasting model itself* propagates the perturbed demand signal to neighboring products in the graph. Products connected by high co-purchase weights (complementary goods) or high correlation weights (co-seasonal items) will exhibit secondary demand effects in the model's predictions, even if they are not directly targeted by the scenario.

This creates a two-tier impact model:

1. **Direct impact**: SKUs in the scenario's target category receive the full demand multiplier $\mu_s$.
2. **Indirect impact**: Graph-adjacent SKUs exhibit attenuated demand effects proportional to their edge weights in $A$, captured automatically by the GNN's forward pass.

This graph-propagated impact analysis is architecturally impossible with standalone temporal models (LSTM, Transformer) and is a primary justification for deploying the TFT–GNN model in production despite its marginally higher baseline WAPE (§7.6 of the forecasting model documentation).

### 8.2 Operational Justification

The adversarial scenario engine directly justifies the TFT–GNN architecture choice. The product graph enables three capabilities that pure temporal models cannot provide:

1. **Substitution effect modeling.** When a product faces stockout under an adversarial scenario, the graph identifies which substitute products will absorb demand (intra-category edges).

2. **Complementary cascade detection.** A demand spike on milk propagates to cereal, bread, and butter through co-purchase edges, enabling proactive inventory pre-positioning for the entire basket.

3. **Category-level stress testing.** Graph convolution ensures that category-wide shocks (e.g., "all Dairy products") produce realistic cross-product demand distributions rather than uniform multipliers.

---

## 9. File Structure

```
ml/adversarial/
├── __init__.py
├── scenario_simulator.py          # Layer 1: Deterministic tensor perturbation engine
├── ai_scenario_generator.py       # Layer 2: Rule-based scenario library (8 scenarios)
├── dynamic_ai_scenarios.py        # Layer 3: LLM-augmented dynamic scenario generation
├── interactive_scenario_ai.py     # Layer 4: Natural-language interactive analysis
├── inventory_risk.py              # Risk quantification framework
├── rebalancing.py                 # Inter-store inventory rebalancing
├── populate_db.py                 # Batch pipeline: Layer 1 → risk eval → database
└── populate_db_ai.py              # Full pipeline: Layers 2–3 → risk eval → database

backend/app/api/
├── adversarial.py                 # FastAPI router (REST endpoints)
└── schemas/adversarial.py         # Pydantic request/response models

backend/app/models/
└── adversarial_risk.py            # SQLAlchemy ORM model (PostgreSQL table)
```

---

## 10. Summary of Contributions

This adversarial scenario generation framework makes the following contributions to the StockSense inventory management system:

1. **Formal stress-testing methodology.** Replaces ad-hoc safety stock heuristics with a systematic, scenario-driven evaluation of supply chain resilience.

2. **Multi-fidelity architecture.** Four progressively sophisticated layers allow the system to operate at different infrastructure levels while maintaining a consistent risk quantification framework.

3. **LLM-augmented scenario design.** Dynamic scenario generation from live database analytics ensures that stress tests remain relevant as the business evolves, rather than becoming stale against fixed disruption templates.

4. **Graph-propagated impact analysis.** Integration with the TFT–GNN's product adjacency matrix enables cross-SKU demand contagion modeling—a capability unique to graph-augmented forecasting architectures.

5. **Deterministic risk quantification.** Closed-form risk metrics (stockout detection, severity, days-of-cover, composite risk score) provide reproducible, interpretable assessments without Monte Carlo sampling overhead.

6. **Operational integration.** Full-stack integration from PyTorch inference through PostgreSQL persistence to RESTful API exposure enables real-time decision support in production environments.

---

*Document version: March 2026*
