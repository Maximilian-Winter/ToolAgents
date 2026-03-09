---
title: Web Research
---

# Web Research

ToolAgents includes web-research style examples built around the knowledge and crawling helpers.

Representative example:

- `examples/agents/gradio/web_research_agent/`

This example combines:

- an OpenAI-compatible chat provider
- the web search integrations under `ToolAgents.knowledge.web_search`
- the crawler integrations under `ToolAgents.knowledge.web_crawler`
- a Gradio chat interface

Optional dependencies:

- install `ToolAgents[search]` for the maintained web-search and crawling stack

These examples are broader than the minimal supported baseline, so validate the optional search/crawler dependencies before using them in production.
