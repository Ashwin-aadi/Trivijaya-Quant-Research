"""Cross-cutting utilities: config, logging, seeding, run manifests, and shared exceptions.

Nothing in here knows anything about equities. Keep it that way — this package is the plumbing
every other package leans on, so a domain dependency here would leak everywhere.
"""
