# Phase 3 application boundary

`TriageApplication` is the structured boundary intended for a future HTTP, desktop,
or Streamlit interface. It provides submit, case lookup, active worklist, override,
override clearing, completion, audit retrieval, and health operations without
exposing model or database exceptions.

Every operation returns an `ApplicationResponse` containing `ok`, a stable `code`,
and structured `data`. Invalid transitions, duplicate identifiers, missing cases,
authorization failures, integrity failures, and inference failures therefore remain
machine-readable without returning filesystem paths, decoder errors, SQL messages,
or stack traces.

Requests must use an explicit supported request type. Case and actor identifiers use
a bounded safe-character contract. Technicians and clinicians may submit cases;
only clinicians may override or complete them; clinicians and auditors may inspect
the audit log. Auditors cannot access the live worklist. These roles are authorization
rules, not authentication: a production transport must authenticate identities before
constructing `ApplicationActor`.

Responses expose model provenance, scores, thresholds, decisions, warnings, and the
workflow outcome needed for a research interface. Persisted filename keys are already
de-identified by the case store. Raw pixels and source paths are never returned.

The health operation validates current case hashes and the complete audit-event hash
chain. A failed integrity check returns only `unhealthy`; diagnostic details remain
inside the trusted process boundary.
