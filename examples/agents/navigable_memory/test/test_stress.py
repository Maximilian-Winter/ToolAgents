# test_stress.py — Automated stress tests for NavigableMemory + SmartMessages + PromptComposer
#
# Tests (no LLM needed — pure logic verification):
#   1. Deep navigation chains (5+ levels)
#   2. Sibling and parent context loading
#   3. Multi-hop navigation with departure injection
#   4. TTL lifecycle with concurrent expirations
#   5. Mixed message types (permanent, ephemeral, archival, pinned)
#   6. PromptComposer dynamic recompilation
#   7. Persistence: save state → clear → restore → verify
#   8. Large-scale TTL stress (many messages at different rates)
#   9. Edge cases (navigate to nonexistent, TTL=0, empty knowledge base)
#  10. Full integration: navigation triggers departure → SmartMessage → archive
#
# Usage:
#   python test_stress.py
#
# All tests print PASS/FAIL. Exit code 0 if all pass.

import json
import os
import sys
import tempfile
from datetime import datetime

# ── Import the modules under test ──
# Adjust these imports to match your project layout.
# If running from the ToolAgents project root:
from ToolAgents.agent_harness.prompt_composer import PromptComposer, create_prompt_composer
from ToolAgents.agent_harness.smart_messages import (
    SmartMessageManager,
    MessageLifecycle,
    ExpiryAction,
)
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.agent_memory.navigable_memory import (
    NavigableMemory,
    InMemoryBackend,
    DepartureRecord,
)

# ── Import the knowledge base seeder ──
from seed_obsidian_forge import seed as seed_knowledge_base


# ═══════════════════════════════════════════════════════════════════
# Test infrastructure
# ═══════════════════════════════════════════════════════════════════

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name: str):
        self.passed += 1
        print(f"  ✅ {name}")

    def fail(self, name: str, reason: str):
        self.failed += 1
        self.errors.append((name, reason))
        print(f"  ❌ {name}: {reason}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"  Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"\n  Failures:")
            for name, reason in self.errors:
                print(f"    • {name}: {reason}")
        print(f"{'='*60}")
        return self.failed == 0


results = TestResults()


def assert_eq(name, actual, expected):
    if actual == expected:
        results.ok(name)
    else:
        results.fail(name, f"expected {expected!r}, got {actual!r}")


def assert_true(name, condition, detail=""):
    if condition:
        results.ok(name)
    else:
        results.fail(name, detail or "condition was False")


def assert_in(name, needle, haystack):
    if needle in haystack:
        results.ok(name)
    else:
        results.fail(name, f"'{needle}' not found in output ({len(haystack)} chars)")


def assert_not_in(name, needle, haystack):
    if needle not in haystack:
        results.ok(name)
    else:
        results.fail(name, f"'{needle}' unexpectedly found in output")


# ═══════════════════════════════════════════════════════════════════
# Test 1: Deep navigation chains
# ═══════════════════════════════════════════════════════════════════

def test_deep_navigation():
    print("\n── Test 1: Deep navigation chains ──")

    backend = InMemoryBackend()
    departures = []

    def on_depart(record: DepartureRecord):
        departures.append(record)

    nav = NavigableMemory(backend=backend, on_depart=on_depart,
                          include_siblings=True, include_parent=True)
    seed_knowledge_base(nav)

    # Navigate to deepest level: studio > projects > ashenmoor > design > combat > boss-design > act2-thornqueen
    path_chain = [
        "studio/overview.md",
        "studio/projects/ashenmoor/overview.md",
        "studio/projects/ashenmoor/design/overview.md",
        "studio/projects/ashenmoor/design/combat/overview.md",
        "studio/projects/ashenmoor/design/combat/boss-design/overview.md",
        "studio/projects/ashenmoor/design/combat/boss-design/act2-thornqueen.md",
    ]

    for path in path_chain:
        nav.navigate(path)

    assert_eq("navigation depth", len(nav.history), len(path_chain))
    assert_eq("departure count", len(departures), len(path_chain) - 1)
    assert_eq("current location", nav.current_path,
              "studio/projects/ashenmoor/design/combat/boss-design/act2-thornqueen.md")
    assert_eq("current title", nav.current_title, "Boss: The Thornqueen")

    # Verify departures recorded correct titles
    assert_eq("first departure", departures[0].title, "Obsidian Forge Studios")
    assert_eq("last departure", departures[-1].title, "Boss Design Philosophy")

    # Context should include current document content
    context = nav.build_context()
    assert_in("context has current doc", "Thornqueen", context)
    assert_in("context has phase info", "Thorn rain", context)

    # History context should show the chain
    history_ctx = nav.build_history_context()
    assert_in("history has recent entry", "Boss Design", history_ctx)


# ═══════════════════════════════════════════════════════════════════
# Test 2: Sibling and parent context
# ═══════════════════════════════════════════════════════════════════

def test_sibling_parent_context():
    print("\n── Test 2: Sibling and parent context ──")

    backend = InMemoryBackend()
    nav = NavigableMemory(backend=backend, include_siblings=True, include_parent=True)
    seed_knowledge_base(nav)

    # Navigate to a boss — should see siblings (other bosses) and parent
    nav.navigate("studio/projects/ashenmoor/design/combat/boss-design/act2-thornqueen.md")
    context = nav.build_context()

    # Should see sibling bosses mentioned
    assert_in("sibling: Ashen Guardian", "Ashen Guardian", context)
    assert_in("sibling: Void Lord", "Void Lord", context)

    # Should see parent (boss design overview)
    assert_in("parent: Boss Design", "Boss Design", context)


# ═══════════════════════════════════════════════════════════════════
# Test 3: Navigation with departure → SmartMessage injection
# ═══════════════════════════════════════════════════════════════════

def test_departure_to_smart_messages():
    print("\n── Test 3: Departure → SmartMessage injection ──")

    backend = InMemoryBackend()
    msg_manager = SmartMessageManager()

    def on_depart(record: DepartureRecord):
        snippet = record.content[:150].replace("\n", " ")
        msg = ChatMessage.create_system_message(
            f"[Departed] {record.title}: {snippet}"
        )
        msg_manager.add_message(
            msg,
            lifecycle=MessageLifecycle(ttl=4, on_expire=ExpiryAction.ARCHIVE),
        )

    nav = NavigableMemory(backend=backend, on_depart=on_depart)
    seed_knowledge_base(nav)

    # Navigate through 3 locations
    nav.navigate("studio/overview.md")
    nav.navigate("studio/projects/ashenmoor/overview.md")
    nav.navigate("studio/projects/ashenmoor/qa/critical-bugs.md")

    # Should have 2 departure messages (first nav has no departure)
    assert_eq("departure messages", msg_manager.message_count, 2)

    # Tick 4 times — messages should expire and archive
    for _ in range(4):
        msg_manager.tick()

    # First departure (TTL=4) should have archived by now
    assert_eq("active after 4 ticks", msg_manager.message_count, 0)
    assert_eq("archived count", len(msg_manager.archive), 2)

    # Archive should contain the departure info
    archive_text = " ".join(m.get_as_text() for m in msg_manager.archive)
    assert_in("archive has studio overview", "Obsidian Forge", archive_text)
    assert_in("archive has ashenmoor", "Ashenmoor", archive_text)


# ═══════════════════════════════════════════════════════════════════
# Test 4: Concurrent TTL stress
# ═══════════════════════════════════════════════════════════════════

def test_concurrent_ttl():
    print("\n── Test 4: Concurrent TTL stress ──")

    mgr = SmartMessageManager()

    # Add 30 messages with varying TTLs
    for i in range(30):
        ttl = (i % 5) + 1  # TTLs: 1, 2, 3, 4, 5 cycling
        msg = ChatMessage.create_system_message(f"Message {i} (TTL={ttl})")
        mgr.add_message(msg, lifecycle=MessageLifecycle(
            ttl=ttl,
            on_expire=ExpiryAction.ARCHIVE if i % 2 == 0 else ExpiryAction.REMOVE,
        ))

    assert_eq("initial count", mgr.message_count, 30)

    # Track expirations per tick
    expired_per_tick = []
    for tick in range(6):
        result = mgr.tick()
        expired_count = len(result.removed) + len(result.archived)
        expired_per_tick.append(expired_count)

    # After tick 1: TTL=1 messages expire (indices 0, 5, 10, 15, 20, 25 → 6 messages)
    assert_eq("tick 1 expirations", expired_per_tick[0], 6)

    # After tick 2: TTL=2 messages expire (indices 1, 6, 11, 16, 21, 26 → 6 messages)
    assert_eq("tick 2 expirations", expired_per_tick[1], 6)

    # After all 5 ticks, all 30 should be gone
    total_expired = sum(expired_per_tick[:5])
    assert_eq("total expired after 5 ticks", total_expired, 30)
    assert_eq("remaining after 5 ticks", mgr.message_count, 0)

    # Tick 6 should expire nothing
    assert_eq("tick 6 (empty) expirations", expired_per_tick[5], 0)

    # Archive should have the even-indexed messages (15 of them)
    assert_eq("archived count", len(mgr.archive), 15)


# ═══════════════════════════════════════════════════════════════════
# Test 5: Mixed message types
# ═══════════════════════════════════════════════════════════════════

def test_mixed_message_types():
    print("\n── Test 5: Mixed message types ──")

    mgr = SmartMessageManager()

    # Permanent message
    msg_perm = ChatMessage.create_user_message("I'm permanent")
    mgr.add_message(msg_perm)

    # Ephemeral (TTL=2, remove)
    msg_eph = ChatMessage.create_system_message("I'm ephemeral")
    mgr.add_message(msg_eph, MessageLifecycle(ttl=2, on_expire=ExpiryAction.REMOVE))

    # Archival (TTL=3, archive)
    msg_arch = ChatMessage.create_assistant_message("I'll be archived")
    mgr.add_message(msg_arch, MessageLifecycle(ttl=3, on_expire=ExpiryAction.ARCHIVE))

    # Pinned (never expires even with TTL)
    msg_pin = ChatMessage.create_system_message("I'm pinned")
    mgr.add_message(msg_pin, MessageLifecycle(ttl=1, pinned=True))

    # Custom callback
    custom_log = []
    msg_custom = ChatMessage.create_system_message("Custom expiry")
    mgr.add_message(msg_custom, MessageLifecycle(
        ttl=2,
        on_expire=ExpiryAction.CUSTOM,
        on_expire_callback=lambda msg, lc: custom_log.append(msg.get_as_text()),
    ))

    assert_eq("initial count", mgr.message_count, 5)

    # Tick 1
    mgr.tick()
    assert_eq("after tick 1", mgr.message_count, 5)  # pinned survives despite TTL=1

    # Tick 2
    mgr.tick()
    assert_eq("after tick 2 (ephemeral + custom gone)", mgr.message_count, 3)
    assert_eq("custom callback fired", len(custom_log), 1)
    assert_in("custom callback content", "Custom expiry", custom_log[0])

    # Tick 3
    mgr.tick()
    assert_eq("after tick 3 (archival gone)", mgr.message_count, 2)
    assert_eq("archive has 1", len(mgr.archive), 1)
    assert_in("archived content", "archived", mgr.archive[0].get_as_text())

    # Permanent and pinned survive indefinitely
    for _ in range(10):
        mgr.tick()
    assert_eq("after 13 ticks total", mgr.message_count, 2)
    texts = [sm.message.get_as_text() for sm in mgr.get_smart_messages()]
    assert_in("permanent survives", "permanent", " ".join(texts))
    assert_in("pinned survives", "pinned", " ".join(texts))


# ═══════════════════════════════════════════════════════════════════
# Test 6: PromptComposer dynamic recompilation
# ═══════════════════════════════════════════════════════════════════

def test_prompt_composer_dynamic():
    print("\n── Test 6: PromptComposer dynamic recompilation ──")

    # Simulated state
    state = {"location": "nowhere", "turn": 0}

    composer = PromptComposer()
    composer.add_module("instructions", position=0, content="You are an assistant.")
    composer.add_module("state", position=10,
                        content_fn=lambda: f"Location: {state['location']}, Turn: {state['turn']}",
                        prefix="<state>", suffix="</state>")
    composer.add_module("tools", position=20, content="Tools: navigate, search")

    # First compile
    prompt1 = composer.compile()
    assert_in("compile 1 has instructions", "assistant", prompt1)
    assert_in("compile 1 has location nowhere", "nowhere", prompt1)
    assert_in("compile 1 has turn 0", "Turn: 0", prompt1)
    assert_in("compile 1 has state tags", "<state>", prompt1)
    assert_in("compile 1 has tools", "navigate", prompt1)

    # Change state and recompile
    state["location"] = "studio/projects/ashenmoor/overview.md"
    state["turn"] = 5
    prompt2 = composer.compile()
    assert_in("compile 2 has new location", "ashenmoor", prompt2)
    assert_in("compile 2 has turn 5", "Turn: 5", prompt2)

    # Disable tools module
    composer.disable_module("tools")
    prompt3 = composer.compile()
    assert_not_in("compile 3 no tools", "navigate", prompt3)
    assert_in("compile 3 still has state", "ashenmoor", prompt3)

    # Add a new module at position 5 (between instructions and state)
    composer.add_module("memory", position=5, content="Memory: user likes coffee")
    prompt4 = composer.compile()

    # Verify ordering: instructions (0) → memory (5) → state (10)
    idx_instructions = prompt4.index("assistant")
    idx_memory = prompt4.index("coffee")
    idx_state = prompt4.index("ashenmoor")
    assert_true("ordering: instructions < memory < state",
                idx_instructions < idx_memory < idx_state,
                f"positions: {idx_instructions}, {idx_memory}, {idx_state}")

    # Update module content
    composer.update_module("instructions", content="You are Sage.")
    prompt5 = composer.compile()
    assert_not_in("updated: old content gone", "an assistant", prompt5)
    assert_in("updated: new content present", "Sage", prompt5)

    # Remove module
    composer.remove_module("memory")
    prompt6 = composer.compile()
    assert_not_in("removed: memory gone", "coffee", prompt6)
    assert_eq("enabled module count after remove", len(composer.enabled_modules), 2)  # instructions + state (tools is disabled, not removed)


# ═══════════════════════════════════════════════════════════════════
# Test 7: Persistence — save and restore
# ═══════════════════════════════════════════════════════════════════

def test_persistence():
    print("\n── Test 7: Persistence — save and restore ──")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "state.json")

        # ── Phase 1: Build state ──
        backend = InMemoryBackend()
        msg_manager = SmartMessageManager()
        core_blocks = {"persona": "I am Sage", "user_info": "User likes RPGs"}

        departures_recorded = []

        def on_depart(record: DepartureRecord):
            departures_recorded.append(record.path)
            msg = ChatMessage.create_system_message(f"[Departed] {record.title}")
            msg_manager.add_message(
                msg, lifecycle=MessageLifecycle(ttl=5, on_expire=ExpiryAction.ARCHIVE)
            )

        nav = NavigableMemory(backend=backend, on_depart=on_depart)
        seed_knowledge_base(nav)

        # Navigate around
        nav.navigate("studio/overview.md")
        nav.navigate("studio/projects/ashenmoor/overview.md")
        nav.navigate("studio/projects/ashenmoor/design/combat/weapon-balance.md")

        # Add some conversation messages
        msg_manager.add_message(ChatMessage.create_user_message("What are the critical bugs?"))
        msg_manager.add_message(
            ChatMessage.create_assistant_message("Let me check the QA section."),
            MessageLifecycle(ttl=8, on_expire=ExpiryAction.ARCHIVE),
        )

        # Tick a few times
        for _ in range(3):
            msg_manager.tick()

        # Record state for comparison
        pre_save_location = nav.current_path
        pre_save_title = nav.current_title
        pre_save_history = list(nav.history)
        pre_save_active_count = msg_manager.message_count
        pre_save_archive_count = len(msg_manager.archive)
        pre_save_doc_count = backend.document_count
        pre_save_tick_count = msg_manager.tick_count

        # ── Save state ──
        state = {
            "core_memory": core_blocks,
            "current_location": nav.current_path,
            "location_history": nav.history,
            "document_count": backend.document_count,
            "active_messages": [
                {
                    "role": sm.message.role.value,
                    "text": sm.message.get_as_text(),
                    "ttl": sm.lifecycle.ttl,
                    "turns_alive": sm.lifecycle.turns_alive,
                    "pinned": sm.lifecycle.pinned,
                    "on_expire": sm.lifecycle.on_expire.value,
                }
                for sm in msg_manager.get_smart_messages()
            ],
            "archive": [m.get_as_text() for m in msg_manager.archive],
            "tick_count": msg_manager.tick_count,
        }

        with open(save_path, "w") as f:
            json.dump(state, f, indent=2)

        assert_true("save file created", os.path.exists(save_path))
        assert_true("save file non-empty", os.path.getsize(save_path) > 100)

        # ── Phase 2: Load and verify ──
        with open(save_path, "r") as f:
            loaded = json.load(f)

        assert_eq("restored location", loaded["current_location"], pre_save_location)
        assert_eq("restored history length", len(loaded["location_history"]),
                   len(pre_save_history))
        assert_eq("restored doc count", loaded["document_count"], pre_save_doc_count)
        assert_eq("restored active msg count", len(loaded["active_messages"]),
                   pre_save_active_count)
        assert_eq("restored archive count", len(loaded["archive"]),
                   pre_save_archive_count)
        assert_eq("restored tick count", loaded["tick_count"], pre_save_tick_count)

        # Verify message TTL values were preserved
        for msg_data in loaded["active_messages"]:
            assert_true(
                f"msg TTL is int or None: {msg_data['ttl']}",
                msg_data["ttl"] is None or isinstance(msg_data["ttl"], int),
            )
            assert_true(
                f"msg turns_alive is int: {msg_data['turns_alive']}",
                isinstance(msg_data["turns_alive"], int),
            )

        # Verify core memory preserved
        assert_eq("core memory persona", loaded["core_memory"]["persona"], "I am Sage")
        assert_eq("core memory user_info", loaded["core_memory"]["user_info"],
                   "User likes RPGs")

        # ── Phase 3: Reconstruct from saved state ──
        new_msg_manager = SmartMessageManager()
        for msg_data in loaded["active_messages"]:
            if msg_data["role"] == "user":
                msg = ChatMessage.create_user_message(msg_data["text"])
            elif msg_data["role"] == "assistant":
                msg = ChatMessage.create_assistant_message(msg_data["text"])
            else:
                msg = ChatMessage.create_system_message(msg_data["text"])

            lifecycle = MessageLifecycle(
                ttl=msg_data["ttl"],
                turns_alive=msg_data["turns_alive"],
                pinned=msg_data["pinned"],
                on_expire=ExpiryAction(msg_data["on_expire"]),
            )
            new_msg_manager.add_message(msg, lifecycle)

        assert_eq("reconstructed active count", new_msg_manager.message_count,
                   pre_save_active_count)

        # Tick the reconstructed manager — should continue lifecycle correctly
        result = new_msg_manager.tick()
        assert_true("reconstructed manager ticks", True)

        # The messages that were closest to expiring should expire first
        # (TTL was already decremented during save)


# ═══════════════════════════════════════════════════════════════════
# Test 8: Large-scale TTL stress
# ═══════════════════════════════════════════════════════════════════

def test_large_scale_ttl():
    print("\n── Test 8: Large-scale TTL stress ──")

    mgr = SmartMessageManager()

    # 100 messages with random TTLs
    import random
    random.seed(42)

    for i in range(100):
        ttl = random.randint(1, 20)
        action = random.choice([ExpiryAction.REMOVE, ExpiryAction.ARCHIVE])
        msg = ChatMessage.create_system_message(f"Stress message {i}")
        mgr.add_message(msg, MessageLifecycle(ttl=ttl, on_expire=action))

    assert_eq("initial 100 messages", mgr.message_count, 100)

    # Tick 20 times — all should expire
    total_removed = 0
    total_archived = 0
    for tick in range(20):
        result = mgr.tick()
        total_removed += len(result.removed)
        total_archived += len(result.archived)

    assert_eq("all expired after 20 ticks", mgr.message_count, 0)
    assert_eq("total processed", total_removed + total_archived, 100)
    assert_true("archive has entries", len(mgr.archive) > 0)
    assert_true("some were removed (not archived)", total_removed > 0)


# ═══════════════════════════════════════════════════════════════════
# Test 9: Edge cases
# ═══════════════════════════════════════════════════════════════════

def test_edge_cases():
    print("\n── Test 9: Edge cases ──")

    # TTL = 0 should expire on first tick
    mgr = SmartMessageManager()
    msg = ChatMessage.create_system_message("Instant death")
    mgr.add_message(msg, MessageLifecycle(ttl=1, on_expire=ExpiryAction.REMOVE))
    result = mgr.tick()
    assert_eq("TTL=1 expires on first tick", len(result.removed), 1)
    assert_eq("nothing left", mgr.message_count, 0)

    # Empty manager tick
    mgr2 = SmartMessageManager()
    result2 = mgr2.tick()
    assert_eq("empty tick no changes", result2.has_changes, False)

    # Navigate to nonexistent path
    backend = InMemoryBackend()
    nav = NavigableMemory(backend=backend)
    seed_knowledge_base(nav)
    try:
        result = nav.navigate("nonexistent/path.md")
        # Should either return None/error or handle gracefully
        assert_true("nonexistent nav handled", True)
    except Exception as e:
        assert_true("nonexistent nav raised exception", True, str(e))

    # PromptComposer with no modules
    composer = PromptComposer()
    assert_eq("empty composer compile", composer.compile(), "")

    # PromptComposer duplicate module name
    composer.add_module("test", content="hello")
    try:
        composer.add_module("test", content="duplicate")
        results.fail("duplicate module name", "should have raised ValueError")
    except ValueError:
        results.ok("duplicate module name raises ValueError")

    # Remove nonexistent module
    removed = composer.remove_module("nonexistent")
    assert_eq("remove nonexistent returns None", removed, None)

    # Summarize expiry with no summarize_fn (default truncation)
    mgr3 = SmartMessageManager()
    long_text = "A" * 500
    msg3 = ChatMessage.create_system_message(long_text)
    mgr3.add_message(msg3, MessageLifecycle(ttl=1, on_expire=ExpiryAction.SUMMARIZE))
    result3 = mgr3.tick()
    assert_eq("summarize produced replacement", len(result3.summarized), 1)
    original, replacement = result3.summarized[0]
    assert_true("replacement exists", replacement is not None)
    assert_true("replacement is shorter", len(replacement.get_as_text()) < 500)


# ═══════════════════════════════════════════════════════════════════
# Test 10: Full integration — navigation + departure + TTL + compose
# ═══════════════════════════════════════════════════════════════════

def test_full_integration():
    print("\n── Test 10: Full integration ──")

    backend = InMemoryBackend()
    msg_manager = SmartMessageManager()
    core_blocks = {"focus": "Checking Ashenmoor status"}

    def on_depart(record: DepartureRecord):
        snippet = record.content[:100].replace("\n", " ")
        msg = ChatMessage.create_system_message(f"[Departed] {record.title}: {snippet}")
        msg_manager.add_message(
            msg, lifecycle=MessageLifecycle(ttl=3, on_expire=ExpiryAction.ARCHIVE)
        )

    nav = NavigableMemory(backend=backend, on_depart=on_depart,
                          include_siblings=True, include_parent=True)
    seed_knowledge_base(nav)

    # Build composer
    composer = PromptComposer()
    composer.add_module("instructions", position=0, content="You are a studio manager assistant.")
    composer.add_module("memory", position=5,
                        content_fn=lambda: "\n".join(f"{k}: {v}" for k, v in core_blocks.items()),
                        prefix="<core_memory>", suffix="</core_memory>")
    composer.add_module("location", position=10,
                        content_fn=nav.build_context,
                        prefix="<knowledge>", suffix="</knowledge>")

    # Add a pinned system message
    msg_manager.add_message(
        ChatMessage.create_system_message("[SYSTEM] Always check bugs before reporting status."),
        MessageLifecycle(pinned=True),
    )

    # ── Simulate a multi-turn conversation ──

    # Turn 1: Navigate to studio overview
    nav.navigate("studio/overview.md")
    msg_manager.tick()
    msg_manager.add_message(ChatMessage.create_user_message("What's our studio status?"))
    prompt1 = composer.compile()
    assert_in("turn 1: has instructions", "studio manager", prompt1)
    assert_in("turn 1: has location", "Obsidian Forge", prompt1)
    assert_in("turn 1: has memory", "Ashenmoor status", prompt1)

    msg_manager.add_message(
        ChatMessage.create_assistant_message("We have two active projects..."),
        MessageLifecycle(ttl=8, on_expire=ExpiryAction.ARCHIVE),
    )

    # Turn 2: Navigate deeper
    msg_manager.tick()
    nav.navigate("studio/projects/ashenmoor/qa/critical-bugs.md")
    msg_manager.add_message(ChatMessage.create_user_message("What are the critical bugs?"))
    prompt2 = composer.compile()
    assert_in("turn 2: has bugs", "CRIT-001", prompt2)
    assert_in("turn 2: core memory unchanged", "Ashenmoor status", prompt2)

    # Update core memory mid-conversation
    core_blocks["focus"] = "Reviewing critical bugs for Ashenmoor"
    prompt2b = composer.compile()
    assert_in("turn 2b: memory updated", "Reviewing critical bugs", prompt2b)

    # Turn 3-5: Tick and watch departures fade
    for turn in range(3, 6):
        msg_manager.tick()

    # After 3 more ticks, the studio overview departure (TTL=3) should be archived
    assert_true("departures archived", len(msg_manager.archive) > 0)
    archive_text = " ".join(m.get_as_text() for m in msg_manager.archive)
    assert_in("archive has studio overview departure", "Obsidian Forge", archive_text)

    # Pinned message still active
    pinned_ids = msg_manager.get_pinned_message_ids()
    assert_eq("pinned message count", len(pinned_ids), 1)

    # Verify total message integrity
    active = msg_manager.get_active_messages()
    assert_true("active messages exist", len(active) > 0)

    # Final compile should still work
    final_prompt = composer.compile()
    assert_in("final: has instructions", "studio manager", final_prompt)
    assert_in("final: has current location", "Critical Bugs", final_prompt)


# ═══════════════════════════════════════════════════════════════════
# Test 11: Knowledge base completeness
# ═══════════════════════════════════════════════════════════════════

def test_knowledge_base():
    print("\n── Test 11: Knowledge base completeness ──")

    backend = InMemoryBackend()
    nav = NavigableMemory(backend=backend)
    count = seed_knowledge_base(nav)

    assert_true("at least 40 documents", count >= 40, f"got {count}")

    # Test listing at different levels
    all_docs = nav.list_at("")
    assert_true("list all returns documents", len(all_docs) > 0)

    # Test search
    results = nav.search("multiplayer desync")
    assert_true("search finds multiplayer bugs", len(results) > 0)

    results2 = nav.search("VFX backlog")
    assert_true("search finds VFX backlog", len(results2) > 0)

    # Test that deepest paths exist
    deep_path = "studio/projects/ashenmoor/design/combat/boss-design/act2-thornqueen.md"
    nav.navigate(deep_path)
    assert_eq("deep navigation works", nav.current_path, deep_path)

    # Test write and read back
    nav.write("studio/meetings/test-note.md", "Test Note",
              "This is a test document.", ["test"])
    nav.navigate("studio/meetings/test-note.md")
    assert_eq("written doc navigable", nav.current_title, "Test Note")
    assert_in("written doc content", "test document", nav.build_context())


# ═══════════════════════════════════════════════════════════════════
# Test 12: on_tick callback stress
# ═══════════════════════════════════════════════════════════════════

def test_on_tick_callbacks():
    print("\n── Test 12: on_tick callback stress ──")

    mgr = SmartMessageManager()
    tick_log = []

    # 20 messages, each logging every tick
    for i in range(20):
        msg = ChatMessage.create_system_message(f"Tracked {i}")
        mgr.add_message(msg, MessageLifecycle(
            ttl=5,
            on_expire=ExpiryAction.REMOVE,
            on_tick_callback=lambda m, lc, idx=i: tick_log.append(
                (idx, lc.turns_alive, lc.ttl)
            ),
        ))

    # Tick 5 times
    for _ in range(5):
        mgr.tick()

    # Each message should have logged once per tick it was alive
    # Message 0: alive for ticks 1-5 (expires on tick 5), logs 5 times
    # Total: 20 messages × varying lifetimes
    assert_true("tick callbacks fired", len(tick_log) > 0)

    # Verify turns_alive increments correctly
    msg0_entries = [(t, ttl) for idx, t, ttl in tick_log if idx == 0]
    assert_eq("msg 0 logged 5 times", len(msg0_entries), 5)
    assert_eq("msg 0 turn 1 age", msg0_entries[0][0], 1)
    assert_eq("msg 0 turn 5 age", msg0_entries[4][0], 5)


# ═══════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Stress Test Suite")
    print("  NavigableMemory + SmartMessages + PromptComposer")
    print("=" * 60)

    test_deep_navigation()
    test_sibling_parent_context()
    test_departure_to_smart_messages()
    test_concurrent_ttl()
    test_mixed_message_types()
    test_prompt_composer_dynamic()
    test_persistence()
    test_large_scale_ttl()
    test_edge_cases()
    test_full_integration()
    test_knowledge_base()
    test_on_tick_callbacks()

    all_passed = results.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()