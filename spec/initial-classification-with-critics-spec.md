# Initial specification

## Goal & Overview
The objective is to extend the existing single-LLM classification program by building a specialized Multi-Agent Debate (MAD) wrapper to improve accuracy on ambiguous inputs. The system evaluates classification uncertainty through multi-pass self-consistency sampling, routing high-consensus samples directly to the output while subjecting low-consensus edge cases to an adversarial debate loop before final adjudication.

# Software Requirements Specification (SRS): Multi-Agent Debate Classifier

## Goal & Overview

The objective is to extend the existing single-LLM classification program by building a specialized Multi-Agent Debate (MAD) wrapper to improve accuracy on ambiguous inputs. The system evaluates classification uncertainty through multi-pass self-consistency sampling, routing high-consensus samples directly to the output while subjecting low-consensus edge cases to an adversarial debate loop before final adjudication.

---

## Inputs, Outputs & Data Contracts

### Inputs

* **Input Text:** The raw text content to be classified.
* **Class Schema:** A user-defined array of categories, including `class_name` and a detailed `description` for each class.
* **Configuration Parameters:**
  * `sampling_runs`: Number of initial classification attempts (Default: `5`).
  * `llm_classifiers`: for each `sampling_runs` attempt, specify which LLM will be used. This is to allow for different classification inputs. If only on LLM is passed, it is assumed to be applied for all the attempts. This can be done via a list or via any other possible structure.
  * `sampling_temperature`: Temperature setting for uncertainty estimation (Default: `0.7`). This is applicable only to LLM calls acepting temperature.
  * `consensus_threshold`: Minimum vote count to bypass debate (Default: `4` out of `5`).
  * `max_debate_rounds`: Hard cap on debate back-and-forth turns (Default: `2`).
  * `llm_debaters`: dictionary or other structure to specifically assign an LLM to a debate role.
  
Given the complexity of some inptus (e.g., `llm_classifiers`, `llm_debaters`), please think about whether a config file may be better. Or allow for both scenarios: the possibility to start with a config file, or to have an "easy" run with just a bunch of CL arguments. Please propose your solution for this.

### Outputs

* **Final Class Label:** The single predicted class name chosen by consensus or adjudication.
* **Execution Path:** Metadata indicating whether the decision was reached via `HIGH_CONSENSUS` or `DEBATE_ADJUDICATED`.
* **Confidence & Voting Breakdown:** Vote distribution across the initial passes (e.g., `{"Class_A": 3, "Class_B": 2}`).
* **Debate Trace (Optional/Debug):** Transcripts of the Critic arguments and Adjudicator reasoning for low-consensus items.

## Core Behavior & Logic

### 1. Sampling & Consensus Assessment
* **Parallel Execution:** Execute `sampling_runs` (e.g., 5) concurrent calls to the base LLM classifier at `sampling_temperature = 0.7`.
* **Vote Aggregation:** Count the resulting predictions across all runs to form a distribution map.
* **Routing Decision:**
  * **High-Consensus Bypass:** If the top-voted class meets or exceeds `consensus_threshold` (e.g., >= 4 votes), bypass the debate loop. Set `execution_path = HIGH_CONSENSUS` and return the top-voted class immediately.
  * **Low-Consensus Escalation:** If no single class meets `consensus_threshold`, escalate the payload to the debate loop. Set `execution_path = DEBATE_ADJUDICATED`.

### 2. Multi-Agent Debate Loop
* **Context Seeding:** Initialize the debate context using the top 2 conflicting candidate classes from the vote distribution, alongside their respective justifications generated during the sampling phase.
* **Critic Agent Interaction:**
  * **Proponent vs. Critic:** Assign a Proponent agent for the leading class candidate and a Critic (Devil's Advocate) agent for the secondary candidate.
  * **Constraint Validation:** Forces both agents to construct arguments strictly referenced against the user-defined class `description` array.
  * **Turn Limit:** Iterate back-and-forth arguments up to `max_debate_rounds`.

### 3. Final Adjudication
* **Decision Pass:** Send the complete debate history to an Adjudicator agent.
* **Evaluation Matrix:** The Adjudicator compares the arguments directly against the original `Input Text` and `Class Schema`.
* **Output Standard:** The Adjudicator assigns the single winning class and provides a concise `adjudicator_reasoning` summary explaining why the winning label strictly fits the user's defined schema.

## Constraints & Guardrails

### 1. Hard Schema Adherence
* **Closed-Vocabulary Outputs:** The final `predicted_class` output by any phase (sampling or adjudication) must strictly match an exact string from the user-provided `classes` list[span_0](start_span)[span_0](end_span). Standard JSON Schema validation must enforce this[span_1](start_span)[span_1](end_span).
* **Fallback Behavior:** If the Adjudicator outputs an invalid label or malformed payload, default automatically to the highest-voted candidate from the initial sampling phase[span_2](start_span)[span_2](end_span).

### 2. Execution & Latency Caps
* **Strict Loop Limits:** The debate loop must hard-cap at `max_debate_rounds` (Default: 2) to prevent infinite back-and-forth loops or token exhaustion[span_3](start_span)[span_3](end_span).
* **Parallel Request Isolation:** All initial sampling calls must run concurrently via async handlers[span_4](start_span)[span_4](end_span). Individual call timeouts should be enforced (e.g., 5 seconds per call) to avoid hung requests dragging down system throughput[span_5](start_span)[span_5](end_span).

### 3. Stateless Design & Integration
* **Plug-and-Play Wrapper:** The module must accept raw payloads and return predictions without maintaining persistent session state, enabling seamless deployment as a microservice on top of the existing single-LLM classification pipeline[span_6](start_span)[span_6](end_span).

### 4. Existing implementation adeherence
Please stick with all the constraints from the existing implementation:
- use pydantic approach as much as possible
- try to implement a multi-query approach per prompt, as already in place
