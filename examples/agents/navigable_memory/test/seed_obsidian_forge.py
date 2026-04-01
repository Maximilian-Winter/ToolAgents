# seed_obsidian_forge.py — Deep knowledge base for stress testing
#
# Obsidian Forge Studios — a mid-size game studio with:
#   - 2 game projects (each with deep sub-trees)
#   - Engineering department with pipelines and infrastructure
#   - Art department with asset pipelines
#   - QA with test plans
#   - People directory
#   - Company operations (finance, legal, HR)
#
# Total: ~80 documents across 5-6 levels of depth.
# Designed to stress-test navigation, TTL, and persistence.

from datetime import datetime


def seed(nav_memory):
    """Seed the full Obsidian Forge Studios knowledge base.

    Returns the count of documents created.
    """
    w = nav_memory.write
    count = 0

    def doc(path, title, content, tags=None):
        nonlocal count
        w(path, title, content, tags or [])
        count += 1

    # ══════════════════════════════════════════════════════════════
    # LEVEL 0-1: Studio overview
    # ══════════════════════════════════════════════════════════════

    doc("studio/overview.md", "Obsidian Forge Studios",
        "# Obsidian Forge Studios\n\n"
        "Mid-size game studio. Founded 2019. ~45 employees.\n"
        "Two active projects, strong engineering culture.\n\n"
        "## Departments\n"
        "- **Ironclad** team — working on 'Ashenmoor' (action RPG)\n"
        "- **Silverwind** team — working on 'Drift Protocol' (racing/combat)\n"
        "- **Core Engineering** — shared tech, build systems, infrastructure\n"
        "- **Art** — concept, 3D, VFX, UI\n"
        "- **QA** — testing, automation, compatibility\n"
        "- **Operations** — finance, HR, legal, marketing\n\n"
        "## Key Dates\n"
        "- Ashenmoor Early Access: August 2026\n"
        "- Drift Protocol announcement: E3 June 2026\n"
        "- Studio anniversary party: November 2026",
        ["studio", "overview"])

    # ══════════════════════════════════════════════════════════════
    # LEVEL 1-5: Project Ashenmoor (action RPG) — deep tree
    # ══════════════════════════════════════════════════════════════

    doc("studio/projects/ashenmoor/overview.md", "Ashenmoor — Project Overview",
        "# Ashenmoor\n\n"
        "Dark fantasy action RPG. Unreal Engine 5.\n"
        "Team Ironclad (18 people).\n\n"
        "## Pillars\n"
        "1. Visceral melee combat with weapon stances\n"
        "2. Interconnected open world with vertical exploration\n"
        "3. Deep crafting tied to world lore\n"
        "4. Emergent NPC behavior via faction system\n\n"
        "## Status\n"
        "- Alpha build: stable\n"
        "- Content: 60% complete\n"
        "- Early Access target: August 2026\n"
        "- Critical path: boss fights, Act 3 zones, multiplayer stability",
        ["ashenmoor", "project", "rpg"])

    # ── Ashenmoor > Design ──

    doc("studio/projects/ashenmoor/design/overview.md", "Ashenmoor Design",
        "# Design Department — Ashenmoor\n\n"
        "Game design, systems design, narrative, level design.\n"
        "Lead: Marcus Chen.",
        ["ashenmoor", "design"])

    doc("studio/projects/ashenmoor/design/combat/overview.md", "Combat System Design",
        "# Combat System\n\n"
        "Stance-based melee with three weapon families:\n"
        "- Heavy (greatswords, hammers) — slow, high stagger\n"
        "- Agile (daggers, rapiers) — fast, combo-focused\n"
        "- Arcane (staves, orbs) — ranged, resource-managed\n\n"
        "Each weapon has 3 stances with distinct movesets.\n"
        "Stance switching is instant and encouraged mid-combo.",
        ["ashenmoor", "combat", "design"])

    doc("studio/projects/ashenmoor/design/combat/stagger-system.md", "Stagger & Poise",
        "# Stagger & Poise System\n\n"
        "Every entity has a Poise bar (hidden from player, shown on bosses).\n"
        "Heavy attacks deal high poise damage. Agile attacks chip.\n\n"
        "## Stagger States\n"
        "1. **Unflinching** — Poise > 50%. No interruption.\n"
        "2. **Staggered** — Poise hits 0. 2-second vulnerability window.\n"
        "3. **Broken** — Staggered 3x in 30s. Extended vulnerability (5s).\n\n"
        "## Tuning Issues\n"
        "- Heavy weapons trivialize stagger on normal enemies\n"
        "- Boss poise recovery too fast for Agile builds\n"
        "- ACTION: Marcus reviewing with combat team this week",
        ["ashenmoor", "combat", "stagger", "tuning"])

    doc("studio/projects/ashenmoor/design/combat/weapon-balance.md", "Weapon Balance Sheet",
        "# Weapon Balance — Current Values\n\n"
        "| Weapon      | DPS (avg) | Stagger/hit | Range | Mobility |\n"
        "|-------------|-----------|-------------|-------|----------|\n"
        "| Greatsword  | 340       | 45          | Med   | Low      |\n"
        "| Warhammer   | 310       | 55          | Short | Low      |\n"
        "| Daggers     | 380       | 12          | Short | High     |\n"
        "| Rapier      | 290       | 18          | Med   | High     |\n"
        "| Fire Staff  | 360       | 8           | Long  | Med      |\n"
        "| Void Orb    | 280       | 15          | Long  | Med      |\n\n"
        "## Known Issues\n"
        "- Daggers DPS too high after crit rework\n"
        "- Void Orb underperforming — needs utility buff\n"
        "- Warhammer feels bad in boss fights (too slow, misses windows)",
        ["ashenmoor", "combat", "balance", "weapons"])

    doc("studio/projects/ashenmoor/design/combat/boss-design/overview.md", "Boss Design Philosophy",
        "# Boss Design\n\n"
        "Bosses are multi-phase encounters with unique mechanics.\n"
        "Each boss tests a different combat skill:\n"
        "- Act 1 bosses: teach stance switching\n"
        "- Act 2 bosses: test positioning and timing\n"
        "- Act 3 bosses: demand full mastery\n\n"
        "## Boss Count\n"
        "- Act 1: 3 bosses (DONE)\n"
        "- Act 2: 4 bosses (3 DONE, 1 IN PROGRESS)\n"
        "- Act 3: 3 bosses (1 DONE, 2 NOT STARTED)\n"
        "- Final boss: NOT STARTED\n"
        "- Optional bosses: 2 (NOT STARTED)",
        ["ashenmoor", "bosses", "design"])

    doc("studio/projects/ashenmoor/design/combat/boss-design/act1-guardian.md",
        "Boss: The Ashen Guardian",
        "# The Ashen Guardian (Act 1, Boss 1)\n\n"
        "Tutorial boss. Teaches stance switching.\n\n"
        "## Phases\n"
        "1. Shield phase — must use Heavy to break shield\n"
        "2. Aggressive phase — must use Agile to dodge-counter\n"
        "3. Arcane phase — ranged attacks, must use Arcane to counter\n\n"
        "## Status: DONE — playtested, tuned, approved\n"
        "## Difficulty: Easy (by design)\n"
        "## Music: 'Embers of the Forgotten' (Jordan, track delivered)",
        ["ashenmoor", "boss", "act1", "done"])

    doc("studio/projects/ashenmoor/design/combat/boss-design/act2-thornqueen.md",
        "Boss: The Thornqueen",
        "# The Thornqueen (Act 2, Boss 3)\n\n"
        "Arena boss with environmental hazards.\n\n"
        "## Phases\n"
        "1. Vine attacks — dodge through gaps, find openings\n"
        "2. Thorn rain — shrinking safe zones, DPS check\n"
        "3. Enrage — all mechanics overlap, heal on player damage\n\n"
        "## Status: IN PROGRESS\n"
        "- Phase 1 & 2: implemented and playable\n"
        "- Phase 3: blocked on VFX (thorn rain particles)\n"
        "- Heal-on-damage mechanic needs engineering support\n\n"
        "## BLOCKED: VFX team overloaded. Estimated delivery: 2 weeks.",
        ["ashenmoor", "boss", "act2", "blocked"])

    doc("studio/projects/ashenmoor/design/combat/boss-design/act3-voidlord.md",
        "Boss: The Void Lord (Act 3 Final)",
        "# The Void Lord (Act 3 Final Boss)\n\n"
        "Multi-stage encounter with phase transitions via cutscene.\n\n"
        "## Concept\n"
        "- Phase 1: humanoid swordfighter (tests all stances)\n"
        "- Phase 2: transforms — arena becomes void space\n"
        "- Phase 3: reality-bending attacks, telegraphed patterns\n\n"
        "## Status: NOT STARTED\n"
        "- Concept doc: APPROVED\n"
        "- Implementation: blocked on Act 3 zone completion\n"
        "- Estimated start: June 2026\n"
        "- Estimated completion: July 2026 (tight for August EA)",
        ["ashenmoor", "boss", "act3", "planned"])

    # ── Ashenmoor > Design > World ──

    doc("studio/projects/ashenmoor/design/world/overview.md", "World Design",
        "# World Design — Ashenmoor\n\n"
        "Interconnected open world, 3 acts, each with 2-3 zones.\n\n"
        "## Zones\n"
        "- Act 1: Ashen Vale, Ironwood Forest (DONE)\n"
        "- Act 2: Drowned Reaches, Thornheart Marsh, Crystal Caverns (DONE)\n"
        "- Act 3: Void Wastes (IN PROGRESS), Obsidian Citadel (NOT STARTED)\n\n"
        "## Vertical Exploration\n"
        "Every zone has above-ground and underground layers.\n"
        "Shortcuts between layers reward exploration.",
        ["ashenmoor", "world", "zones"])

    doc("studio/projects/ashenmoor/design/world/act3-void-wastes.md", "Act 3: Void Wastes",
        "# Void Wastes — Act 3 Zone\n\n"
        "Reality-fractured desert. Environmental storytelling heavy.\n\n"
        "## Layout\n"
        "- Entry from Crystal Caverns (elevator sequence)\n"
        "- Central hub: Ruined Observatory\n"
        "- 4 sub-areas radiating from hub\n"
        "- Underground: Collapsed Tunnels (optional area)\n\n"
        "## Status: IN PROGRESS (40% complete)\n"
        "- Hub and 2 sub-areas playable\n"
        "- 2 sub-areas: greyboxed only\n"
        "- Underground: not started\n"
        "- Art pass needed on completed areas",
        ["ashenmoor", "world", "act3", "inprogress"])

    # ── Ashenmoor > Design > Narrative ──

    doc("studio/projects/ashenmoor/design/narrative/overview.md", "Narrative Design",
        "# Narrative — Ashenmoor\n\n"
        "Writer: Elena Vasquez (contract, remote).\n\n"
        "## Story Arc\n"
        "Act 1: Discovery — player learns the world is dying\n"
        "Act 2: Descent — corruption spreads, factions fracture\n"
        "Act 3: Convergence — choose allegiance, confront the Void\n\n"
        "## Dialogue System\n"
        "Bark-based (no dialogue trees). NPCs react to faction standing.\n"
        "~2,400 lines written. ~800 remaining.\n\n"
        "## Lore Entries\n"
        "128 lore items placed in world. 64 written, 64 placeholder.",
        ["ashenmoor", "narrative", "writing"])

    doc("studio/projects/ashenmoor/design/narrative/factions.md", "Faction System",
        "# Factions\n\n"
        "Three factions, each with a philosophy:\n"
        "1. **Iron Covenant** — preserve order through strength\n"
        "2. **Verdant Circle** — harmony with nature, reject technology\n"
        "3. **Void Seekers** — embrace the corruption for power\n\n"
        "## Mechanics\n"
        "- Standing: -100 to +100 per faction\n"
        "- Actions shift standing (quests, kills, dialogue choices)\n"
        "- Shops, areas, and endings locked behind standing thresholds\n"
        "- Can't max all three — forces meaningful choice\n\n"
        "## Act 3 Impact\n"
        "Highest standing faction determines Act 3 variant:\n"
        "- Iron: storm the Citadel with army\n"
        "- Verdant: sneak in through nature paths\n"
        "- Void: walk in as ally, betray from within",
        ["ashenmoor", "factions", "narrative"])

    # ── Ashenmoor > Engineering ──

    doc("studio/projects/ashenmoor/engineering/overview.md", "Ashenmoor Engineering",
        "# Engineering — Ashenmoor\n\n"
        "Lead: Priya Sharma. 6 engineers.\n\n"
        "## Tech Stack\n"
        "- Unreal Engine 5.4\n"
        "- Custom gameplay framework on top of GAS\n"
        "- Dedicated server for multiplayer (optional co-op)\n"
        "- CI/CD: Jenkins → SteamPipe\n\n"
        "## Hot Issues\n"
        "1. Multiplayer desync on boss phase transitions\n"
        "2. Memory leak in particle system (VFX heavy areas)\n"
        "3. Load times in Act 2 zones (streaming not optimized)",
        ["ashenmoor", "engineering", "tech"])

    doc("studio/projects/ashenmoor/engineering/multiplayer.md", "Multiplayer Architecture",
        "# Multiplayer — Co-op (2 players)\n\n"
        "Optional drop-in co-op. Host-authoritative.\n\n"
        "## Architecture\n"
        "- UE5 replication with custom prediction\n"
        "- Boss state machine replicated via RPC\n"
        "- Loot: instanced per player\n\n"
        "## Known Bugs\n"
        "- CRIT: Phase transition desync — client sees wrong phase\n"
        "- HIGH: Stagger state not replicated reliably\n"
        "- MED: Cosmetic desync on stance switch animation\n\n"
        "## Owner: Jake Torres\n"
        "## ETA for CRIT fix: 1 week",
        ["ashenmoor", "multiplayer", "networking", "bugs"])

    doc("studio/projects/ashenmoor/engineering/performance.md", "Performance Budget",
        "# Performance Targets\n\n"
        "## Minimum Spec\n"
        "- 1080p 30fps on GTX 1060 / RX 580\n"
        "- 8GB RAM, SSD recommended\n\n"
        "## Current State\n"
        "- Act 1 zones: hitting targets ✅\n"
        "- Act 2 zones: 24-28fps in Thornheart Marsh ❌\n"
        "- Act 3 zones: untested\n\n"
        "## Main Bottlenecks\n"
        "1. Particle overdraw in VFX-heavy areas\n"
        "2. Streaming not aggressive enough (texture pop-in)\n"
        "3. GC spikes from temporary actors in combat",
        ["ashenmoor", "performance", "optimization"])

    # ── Ashenmoor > Art ──

    doc("studio/projects/ashenmoor/art/overview.md", "Ashenmoor Art",
        "# Art — Ashenmoor\n\n"
        "Art Director: Sofia Rodriguez. 5 artists + 2 freelancers.\n\n"
        "## Style\n"
        "Dark fantasy, painterly textures, volumetric lighting.\n"
        "Inspired by Berserk manga and Gothic architecture.\n\n"
        "## Pipeline\n"
        "Concept → Blockout → High Poly → Game-ready → Texture → Lighting pass\n\n"
        "## Current Load\n"
        "- Character models: 85% complete\n"
        "- Environment assets: 70% complete\n"
        "- VFX: 50% complete (bottleneck)\n"
        "- UI: 40% complete",
        ["ashenmoor", "art", "pipeline"])

    doc("studio/projects/ashenmoor/art/vfx-backlog.md", "VFX Backlog",
        "# VFX Backlog — Priority Order\n\n"
        "VFX artist: ONE person (Kai). Major bottleneck.\n\n"
        "1. ⬜ Thornqueen thorn rain (BLOCKING boss) — 3 days\n"
        "2. ⬜ Void Wastes ambient particles — 2 days\n"
        "3. ⬜ Weapon enchantment effects (3 types) — 4 days\n"
        "4. ⬜ Boss death dissolve shader — 2 days\n"
        "5. ⬜ Act 3 portal effects — 3 days\n"
        "6. ⬜ UI hit feedback particles — 1 day\n"
        "7. ⬜ Crafting forge fire — 1 day\n\n"
        "## Total: ~16 work days. Kai available: 12 days before EA.\n"
        "## RISK: Cannot complete all. Must cut items 5-7 or get help.",
        ["ashenmoor", "vfx", "backlog", "risk"])

    # ── Ashenmoor > QA ──

    doc("studio/projects/ashenmoor/qa/overview.md", "Ashenmoor QA",
        "# QA — Ashenmoor\n\n"
        "QA Lead: Sam Park. 3 testers.\n\n"
        "## Test Coverage\n"
        "- Act 1: full regression, stable ✅\n"
        "- Act 2: in progress, 2 blockers found\n"
        "- Act 3: not testable yet\n"
        "- Multiplayer: sporadic testing, needs dedicated pass\n\n"
        "## Bug Stats (last 30 days)\n"
        "- Opened: 47\n"
        "- Closed: 31\n"
        "- Critical open: 3\n"
        "- High open: 8",
        ["ashenmoor", "qa", "testing"])

    doc("studio/projects/ashenmoor/qa/critical-bugs.md", "Critical Bugs",
        "# Critical Bugs — Ashenmoor\n\n"
        "## CRIT-001: Multiplayer phase desync\n"
        "Owner: Jake Torres (Engineering)\n"
        "Repro: Host triggers boss phase 2, client stays in phase 1.\n"
        "Impact: Co-op unplayable for boss fights.\n"
        "ETA: 1 week.\n\n"
        "## CRIT-002: Memory leak in Thornheart Marsh\n"
        "Owner: Priya Sharma (Engineering)\n"
        "Repro: Play in Thornheart for >20 min, RAM climbs to 12GB+.\n"
        "Impact: Crash on min-spec machines.\n"
        "ETA: investigating.\n\n"
        "## CRIT-003: Save corruption on faction standing overflow\n"
        "Owner: Jake Torres (Engineering)\n"
        "Repro: Standing exceeds 100 via quest stacking exploit.\n"
        "Impact: Save file becomes unloadable.\n"
        "ETA: 3 days (needs clamp + migration script).",
        ["ashenmoor", "bugs", "critical"])

    # ══════════════════════════════════════════════════════════════
    # LEVEL 1-4: Project Drift Protocol (racing/combat)
    # ══════════════════════════════════════════════════════════════

    doc("studio/projects/drift-protocol/overview.md", "Drift Protocol — Project Overview",
        "# Drift Protocol\n\n"
        "Futuristic racing/combat game. Unreal Engine 5.\n"
        "Team Silverwind (12 people).\n\n"
        "## Concept\n"
        "Anti-gravity racing with vehicular combat.\n"
        "Think WipEout meets Twisted Metal.\n\n"
        "## Status\n"
        "- Pre-alpha prototype\n"
        "- 3 tracks playable\n"
        "- 5 vehicles implemented\n"
        "- E3 announcement demo: June 2026\n"
        "- Vertical slice deadline: May 15, 2026",
        ["drift", "project", "racing"])

    doc("studio/projects/drift-protocol/design/overview.md", "Drift Protocol Design",
        "# Design — Drift Protocol\n\n"
        "Lead Designer: Alex Kim.\n\n"
        "## Core Loop\n"
        "Race → earn credits → upgrade vehicle → race harder tracks\n"
        "Combat is integrated: weapons drop on track, PvP elimination mode.\n\n"
        "## Modes\n"
        "1. Grand Prix (3-race series)\n"
        "2. Arena (combat deathmatch)\n"
        "3. Time Trial (leaderboards)\n"
        "4. Career (story-driven progression)",
        ["drift", "design"])

    doc("studio/projects/drift-protocol/design/tracks/overview.md", "Track Design",
        "# Tracks — Drift Protocol\n\n"
        "## Completed\n"
        "1. Neon Circuit — beginner track, city environment\n"
        "2. Volcanic Drift — intermediate, environmental hazards\n"
        "3. Orbital Station — advanced, zero-G sections\n\n"
        "## In Design\n"
        "4. Abyssal Trench — underwater tunnel, low visibility\n"
        "5. Cloudspire — vertical track, extreme altitude changes\n\n"
        "## Needed for E3\n"
        "Tracks 1-3 polished. Track 4 greybox for demo variety.",
        ["drift", "tracks", "level-design"])

    doc("studio/projects/drift-protocol/design/vehicles.md", "Vehicle Design",
        "# Vehicles — Drift Protocol\n\n"
        "5 vehicle classes:\n"
        "| Class     | Speed | Armor | Weapon Slots | Special        |\n"
        "|-----------|-------|-------|-------------|----------------|\n"
        "| Striker   | High  | Low   | 2           | Afterburner    |\n"
        "| Tank      | Low   | High  | 3           | Shield ram     |\n"
        "| Phantom   | Med   | Low   | 1           | Cloak (3s)     |\n"
        "| Engineer  | Med   | Med   | 2           | Deploy turret  |\n"
        "| Hybrid    | Med   | Med   | 2           | Mode switch    |\n\n"
        "## Balance Status\n"
        "Phantom cloak is OP in Arena mode. Needs cooldown nerf or detection counter.",
        ["drift", "vehicles", "balance"])

    doc("studio/projects/drift-protocol/engineering/overview.md", "Drift Protocol Engineering",
        "# Engineering — Drift Protocol\n\n"
        "Lead: Daniel Okafor. 4 engineers.\n\n"
        "## Tech\n"
        "- UE5 Chaos physics for vehicle simulation\n"
        "- Rollback netcode for competitive multiplayer\n"
        "- Procedural track segments (stretch goal)\n\n"
        "## E3 Demo Risks\n"
        "1. Netcode not stable enough for live demo\n"
        "2. Loading transitions between tracks too slow\n"
        "3. Need fallback plan: local multiplayer only for E3",
        ["drift", "engineering"])

    doc("studio/projects/drift-protocol/art/overview.md", "Drift Protocol Art",
        "# Art — Drift Protocol\n\n"
        "Art Director: Sofia Rodriguez (shared with Ashenmoor).\n\n"
        "## Style\n"
        "Neon-drenched cyberpunk. High contrast. Synthwave palette.\n\n"
        "## Status\n"
        "- Vehicle models: 5/5 done, 3/5 textured\n"
        "- Track art: Neon Circuit polished, others greybox\n"
        "- UI: placeholder only\n"
        "- RISK: Sofia split between two projects",
        ["drift", "art"])

    # ══════════════════════════════════════════════════════════════
    # Core Engineering (shared tech)
    # ══════════════════════════════════════════════════════════════

    doc("studio/engineering/overview.md", "Core Engineering",
        "# Core Engineering\n\n"
        "Shared technology and infrastructure team.\n"
        "Lead: Priya Sharma (also Ashenmoor engineering lead).\n\n"
        "## Responsibilities\n"
        "- Build system and CI/CD\n"
        "- Shared engine plugins\n"
        "- DevOps and infrastructure\n"
        "- Code review standards\n"
        "- Performance profiling tools",
        ["engineering", "shared"])

    doc("studio/engineering/ci-cd.md", "CI/CD Pipeline",
        "# CI/CD Pipeline\n\n"
        "## Stack\n"
        "- Jenkins (self-hosted, 4 build agents)\n"
        "- Perforce for version control\n"
        "- SteamPipe for distribution\n\n"
        "## Build Times\n"
        "- Ashenmoor full build: 45 min (needs optimization)\n"
        "- Drift Protocol full build: 20 min\n"
        "- Incremental: 5-8 min both projects\n\n"
        "## Issues\n"
        "- Build agent 3 keeps dying (hardware issue, new machine ordered)\n"
        "- Perforce workspace configs inconsistent across team\n"
        "- No automated testing in pipeline yet (QA wants this)",
        ["engineering", "cicd", "devops"])

    doc("studio/engineering/coding-standards.md", "Coding Standards",
        "# Coding Standards\n\n"
        "## C++ (UE5)\n"
        "- Epic coding standard with studio additions\n"
        "- UCLASS/UPROPERTY for all gameplay code\n"
        "- No raw pointers in gameplay code (use TSharedPtr/TWeakPtr)\n"
        "- Blueprint-exposable where designers need access\n\n"
        "## Code Review\n"
        "- All changes require 1 reviewer\n"
        "- Critical systems (netcode, save): 2 reviewers\n"
        "- Review turnaround target: 24 hours",
        ["engineering", "standards"])

    # ══════════════════════════════════════════════════════════════
    # People directory
    # ══════════════════════════════════════════════════════════════

    doc("studio/people/overview.md", "People Directory",
        "# People — Obsidian Forge Studios\n\n"
        "45 employees across 6 departments.\n"
        "Key people listed in individual entries.",
        ["people"])

    doc("studio/people/marcus-chen.md", "Marcus Chen",
        "# Marcus Chen — Lead Designer (Ashenmoor)\n\n"
        "At studio since founding. 15 years industry experience.\n"
        "Previously at Naughty Dog (combat design on Uncharted 4).\n\n"
        "## Strengths\n"
        "- Exceptional combat feel intuition\n"
        "- Good at rapid prototyping\n\n"
        "## Current Focus\n"
        "- Stagger system tuning (this week)\n"
        "- Act 3 boss concepts (next two weeks)\n\n"
        "## Notes\n"
        "Going on vacation June 10-20. Plan around this.",
        ["people", "design", "ashenmoor"])

    doc("studio/people/priya-sharma.md", "Priya Sharma",
        "# Priya Sharma — Engineering Lead\n\n"
        "Dual role: Core Engineering lead + Ashenmoor engineering lead.\n"
        "10 years experience. Previously at Epic Games.\n\n"
        "## Strengths\n"
        "- Deep UE5 internals knowledge\n"
        "- Performance optimization expert\n\n"
        "## Concern\n"
        "Overloaded with dual role. Has flagged this to management.\n"
        "Proposal: promote Jake Torres to Ashenmoor eng lead after EA.",
        ["people", "engineering"])

    doc("studio/people/sofia-rodriguez.md", "Sofia Rodriguez",
        "# Sofia Rodriguez — Art Director\n\n"
        "Shared across both projects.\n\n"
        "## Concern\n"
        "Split attention between Ashenmoor and Drift Protocol.\n"
        "Ashenmoor needs art polish pass, Drift needs art direction.\n"
        "Proposed: hire second art lead for Drift Protocol.",
        ["people", "art"])

    doc("studio/people/jake-torres.md", "Jake Torres",
        "# Jake Torres — Senior Engineer (Ashenmoor)\n\n"
        "3 years at studio. Networking specialist.\n\n"
        "## Current Load\n"
        "- CRIT-001: Multiplayer phase desync (this week)\n"
        "- CRIT-003: Save corruption (after CRIT-001)\n"
        "- Candidate for Ashenmoor eng lead after EA\n\n"
        "## Notes\n"
        "Very reliable. Priya trusts him with critical fixes.",
        ["people", "engineering", "ashenmoor"])

    doc("studio/people/kai-nakamura.md", "Kai Nakamura",
        "# Kai Nakamura — VFX Artist\n\n"
        "ONLY VFX artist at the studio. Major bottleneck.\n\n"
        "## Current Queue\n"
        "See: studio/projects/ashenmoor/art/vfx-backlog.md\n"
        "16 days of work, 12 days available.\n\n"
        "## Risk\n"
        "If Kai gets sick or leaves, VFX pipeline stops entirely.\n"
        "HR exploring contract VFX artist as backup.",
        ["people", "art", "vfx", "risk"])

    doc("studio/people/alex-kim.md", "Alex Kim",
        "# Alex Kim — Lead Designer (Drift Protocol)\n\n"
        "2 years at studio. Previously at EA (Need for Speed team).\n\n"
        "## Focus\n"
        "- E3 demo build (all-consuming until June)\n"
        "- Vehicle balance pass\n"
        "- Track 4 greybox\n\n"
        "## Style\n"
        "Prefers rapid iteration over documentation.\n"
        "Sometimes forgets to update design docs — nudge gently.",
        ["people", "design", "drift"])

    doc("studio/people/elena-vasquez.md", "Elena Vasquez",
        "# Elena Vasquez — Narrative Designer (Contract)\n\n"
        "Remote contractor. Writing all Ashenmoor dialogue and lore.\n"
        "Contract runs through September 2026.\n\n"
        "## Delivery\n"
        "- 2,400/3,200 lines delivered\n"
        "- Remaining 800 lines due by July\n"
        "- Quality: consistently excellent\n"
        "- Communication: weekly Friday check-in via Zoom",
        ["people", "narrative", "contract"])

    doc("studio/people/sam-park.md", "Sam Park",
        "# Sam Park — QA Lead\n\n"
        "5 years at studio. Manages 3 testers.\n\n"
        "## Advocacy\n"
        "Pushing for automated testing in CI/CD pipeline.\n"
        "Has written proposal, needs engineering bandwidth.\n\n"
        "## Current Priority\n"
        "Act 2 regression for Ashenmoor. Multiplayer test pass next.",
        ["people", "qa"])

    doc("studio/people/daniel-okafor.md", "Daniel Okafor",
        "# Daniel Okafor — Engineering Lead (Drift Protocol)\n\n"
        "2 years at studio. Networking and physics specialist.\n\n"
        "## Focus\n"
        "- Rollback netcode stability for E3 demo\n"
        "- Vehicle physics tuning\n"
        "- Preparing fallback plan (local-only demo)\n\n"
        "## Risk\n"
        "E3 demo netcode might not be ready. Has contingency.",
        ["people", "engineering", "drift"])

    # ══════════════════════════════════════════════════════════════
    # Operations
    # ══════════════════════════════════════════════════════════════

    doc("studio/operations/overview.md", "Operations",
        "# Operations — Obsidian Forge Studios\n\n"
        "Finance, HR, legal, marketing.",
        ["operations"])

    doc("studio/operations/finance/overview.md", "Finance",
        "# Finance Overview\n\n"
        "## Runway\n"
        "Current runway: 14 months (through August 2027).\n"
        "Revenue dependent on Ashenmoor EA sales.\n\n"
        "## Monthly Burn\n"
        "~$280K/month (salaries + contractors + infrastructure).\n\n"
        "## Ashenmoor Revenue Projection\n"
        "Conservative: $800K first 3 months of EA\n"
        "Moderate: $1.5M first 3 months\n"
        "Optimistic: $3M first 3 months\n\n"
        "Need moderate scenario to fund Drift Protocol through 2027.",
        ["operations", "finance"])

    doc("studio/operations/finance/contractor-budget.md", "Contractor Budget",
        "# Contractor Budget — Q2 2026\n\n"
        "| Contractor     | Role           | Monthly  | End Date |\n"
        "|----------------|----------------|----------|----------|\n"
        "| Elena Vasquez   | Narrative      | $8,000   | Sep 2026 |\n"
        "| Mika (pixel art)| Environment art| $6,400   | Jun 2026 |\n"
        "| Jordan (music)  | Soundtrack     | $5,000   | May 2026 |\n"
        "| TBD (VFX)       | VFX backup     | ~$7,000  | TBD      |\n\n"
        "## Total contractor spend: ~$26,400/month\n"
        "## Budget remaining Q2: $45,000 unallocated\n"
        "NOTE: VFX contractor hire would consume most of remaining budget.",
        ["operations", "finance", "contractors"])

    doc("studio/operations/hr/hiring.md", "Hiring Plan",
        "# Hiring Plan\n\n"
        "## Open Positions\n"
        "1. VFX Artist (contract) — URGENT, Kai is sole VFX person\n"
        "2. Art Lead for Drift Protocol — after E3\n"
        "3. Senior Engineer — backfill if Jake promoted\n\n"
        "## Pipeline\n"
        "- VFX: 3 candidates in pipeline, interviewing this week\n"
        "- Art Lead: job posting draft ready, waiting for E3\n"
        "- Senior Eng: not posted yet",
        ["operations", "hr", "hiring"])

    doc("studio/operations/marketing/ashenmoor-plan.md", "Ashenmoor Marketing Plan",
        "# Marketing — Ashenmoor\n\n"
        "## EA Launch Plan\n"
        "- Trailer: needs new footage (Act 2 bosses) — due July 1\n"
        "- Steam page update: existing, needs new screenshots\n"
        "- Press kit: 60% done\n"
        "- Streamer outreach: list of 50 targets, emails drafted\n"
        "- Launch discount: 15% for first week\n\n"
        "## Timeline\n"
        "- July 1: trailer and press kit ready\n"
        "- July 15: send to press/streamers\n"
        "- August 1: EA launch\n"
        "- August 1-7: launch week support + community management",
        ["operations", "marketing", "ashenmoor"])

    doc("studio/operations/marketing/drift-e3.md", "Drift Protocol E3 Plan",
        "# E3 Plan — Drift Protocol\n\n"
        "## Demo\n"
        "- 3-minute gameplay trailer (pre-rendered + gameplay)\n"
        "- Playable demo on show floor (Neon Circuit track)\n"
        "- 2 vehicles playable in demo\n\n"
        "## Booth\n"
        "- 10x10 booth in indie section\n"
        "- 4 demo stations\n"
        "- Staff: Alex + Daniel + 2 others\n\n"
        "## Risk\n"
        "If netcode isn't ready: local split-screen only.\n"
        "Trailer must be done by May 25 (submission deadline).\n\n"
        "## Budget: $35,000 (booth + travel + swag)",
        ["operations", "marketing", "drift", "e3"])

    # ══════════════════════════════════════════════════════════════
    # Meeting notes (simulates real usage)
    # ══════════════════════════════════════════════════════════════

    doc("studio/meetings/2026-03-24-standup.md", "Standup 2026-03-24",
        "# Daily Standup — March 24, 2026\n\n"
        "## Ashenmoor\n"
        "- Marcus: finishing stagger tuning pass, results by Wednesday\n"
        "- Jake: CRIT-001 root cause found, fix in progress\n"
        "- Kai: starting Thornqueen VFX today\n"
        "- Sam: Act 2 regression found 2 new bugs (filed)\n\n"
        "## Drift Protocol\n"
        "- Alex: vehicle balance spreadsheet updated\n"
        "- Daniel: rollback netcode test passed for 2-player, 4-player failing\n\n"
        "## Blockers\n"
        "- Kai needs Phase 3 design doc from Marcus before starting VFX\n"
        "- Daniel needs test environment for 4-player netcode\n\n"
        "## Action Items\n"
        "- Marcus → deliver Phase 3 doc to Kai by EOD Tuesday\n"
        "- IT → set up 4-player test environment by Wednesday",
        ["meeting", "standup"])

    doc("studio/meetings/2026-03-21-leadership.md", "Leadership Meeting 2026-03-21",
        "# Leadership Meeting — March 21, 2026\n\n"
        "## Attendees\n"
        "CEO, Marcus, Priya, Sofia, Alex, Daniel, Sam, HR lead\n\n"
        "## Key Decisions\n"
        "1. VFX contractor hire approved — HR to fast-track\n"
        "2. Ashenmoor EA date confirmed: August 1\n"
        "3. E3 budget approved at $35K\n"
        "4. If netcode isn't ready by May 10: local-only E3 demo\n\n"
        "## Concerns Raised\n"
        "- Priya: dual role unsustainable, proposed Jake promotion\n"
        "- Sofia: can't art-direct two projects well, needs help\n"
        "- Sam: no automated testing is increasing bug escape rate\n\n"
        "## Follow-ups\n"
        "- CEO to discuss Jake promotion with board\n"
        "- HR to post Art Lead position after E3\n"
        "- Priya + Sam to scope automated testing MVP",
        ["meeting", "leadership"])

    print(f"  Seeded {count} documents into Obsidian Forge knowledge base.")
    return count
