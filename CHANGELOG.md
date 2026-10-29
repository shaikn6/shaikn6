# Changelog

## [3.2.0] - 2026-10-29
### Changed
- Profile reads as a landing page for LLM, generative AI and agentic AI roles: plainer headline, three
  "Start here" projects (llm-gateway, ledger-service, lethe) with one checked result each, then more work.
- Skills are grouped by evidence; each group links to the repo that shows it. Skills with no public repo
  behind them were dropped from this list.
- Merged upstream count shown (20, checked against GitHub); a plain contact line in the footer.

## [3.1.0] - 2026-10-17
### Changed
- Profile now leads with generative AI for fintech: RAG pipelines, agentic workflows and AWS ML platforms
  (header text, intro, "Now" line with the advisory credit, stack line with AWS and platform tools).
- lethe row uses the same measured results as the project README (0.611 vs 0.641, then 0.707).

## [3.0.0] - 2026-10-15
### Changed
- Refocused the profile on LLM, generative-AI and agentic AI work: new header text, featured
  LLM and agent projects only (adds lethe), LLM-only stack line, corrected LinkedIn link.
- Merged list now shows all 15 merged upstream pull requests.
### Removed
- Unused `stack-diagram.svg` from the earlier data-engineering positioning.

## [2.2.0] - 2026-10-11
### Changed
- Header is now a self-hosted animated SVG (`assets/header-*.svg`, dark and light) instead of a
  third-party image service.
- Added a "Now" line, a collapsible "More projects" table and a collapsible "Activity" section
  (self-generated activity graph plus the snake).
- Refreshed the Featured rows and the Open source section to match the repositories today.

## [2.1.0] - 2026-08-28
### Changed
- Cut the README ~65%. Removed the typing SVG, the duplicated certifications table,
  the architecture diagram, the activity-graph card, the 4-tier Featured Work
  structure, and the 8-row tech-stack table.
- Flat 6-repo Featured table (matches the pinned repos), a 3-sentence About, a
  2-line stack, snake only.

## [2.0.0] - 2026-08-28
### Changed
- Repositioned around **AI / LLM Systems Engineer for fintech** (was healthcare + fintech dual-track).
- New **Featured Work** structure: Production Services (ledger-service, llm-gateway,
  fintech-devsecops-pipeline) lead, then LLM Engineering & Agents, Applied ML, and a single
  Regulated-Domain Rigor entry.
- Tech Stack reordered — LLM, Agents, and Backend first; added Go, concurrency control, govulncheck.
- Trimmed the standalone healthcare/clinical-ML section from the profile.

## [1.0.0] - 2026-06-17
### Added
- GitHub profile README with personal bio and project highlights
- Pinned repositories showcasing ML, LLMOps, and data engineering work
- Skills and technology stack overview section
- Contact and social links for professional networking
- Dynamic GitHub stats badges and contribution graph
