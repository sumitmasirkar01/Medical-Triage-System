# Legacy research code

This directory is quarantined legacy code. It contains historical prototypes,
synthetic experiments, and mathematical ideas that have not passed the supported
artifact and evaluation gates.

New application and evaluation code belongs in `stat_triage/`. The supported
package must not import this legacy package; this boundary is enforced by
`tests/test_supported_boundary.py`. Individual algorithms may be migrated only
after their assumptions, target contract, provenance, and tests are documented.

