# Phase 3 persistent case orchestration

`PersistentCaseStore` and `PersistentTriageCoordinator` make the Phase 3 triage
workflow restart-safe. The store uses SQLite transactions and records successful,
unsupported, and failed submissions. Only successful outcomes with immutable model
evidence enter the active priority worklist; failed and unsupported submissions are
retained with `REJECTED` status for audit.

Cases persist the `TriageOutcome`, its model evidence and provenance hashes, workflow
status, arrival time, override state, and completion time. Raw images, DICOM files,
source paths, filenames, and pixel data are not copied into the database. Input hash
keys are replaced with deterministic ordinal aliases while their SHA-256 values are
retained for provenance.

Every mutation appends an audit event. Events form a global SHA-256 hash chain, and
each current case row has its own integrity hash. `validate_integrity()` verifies both
the current case records and the complete ordered event chain. These hashes provide
tamper evidence; they are not digital signatures and do not protect a database when
an attacker can rewrite both records and hashes.

Clinical overrides require a named actor and reason. Completed and rejected cases
cannot be overridden. Priority remains modality-safe: locked decisions determine
the automatic zone, overrides determine the effective zone, aging is recomputed at
read time, and model scores are never compared across modalities.

The implementation is suitable for a local research demonstration. Production use
would additionally require authenticated identities, authorization, encryption,
managed backups, retention policy, database access controls, and external audit-log
anchoring.
