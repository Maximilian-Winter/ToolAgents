"""Reference graph operations for NavigableMemory."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .models import Reference, RefType


class NavigableReferenceMixin:
    def add_reference(self, from_path: str, to_path: str,
                      ref_type: str = RefType.LINKS_TO,
                      note: str = "") -> bool:
        """Create a directed reference between two documents."""
        if not self._has_references_support():
            return False
        return self.backend.add_reference(  # type: ignore[attr-defined]
            from_path, to_path, ref_type, note,
        )

    def remove_reference(self, from_path: str, to_path: str,
                         ref_type: Optional[str] = None) -> int:
        """Remove a reference. Returns number of edges removed."""
        if not self._has_references_support():
            return 0
        return self.backend.remove_reference(  # type: ignore[attr-defined]
            from_path, to_path, ref_type,
        )

    def references_from(self, path: str) -> List[Reference]:
        """List outgoing references from a document."""
        if not self._has_references_support():
            return []
        return self.backend.list_references_from(path)  # type: ignore[attr-defined]

    def references_to(self, path: str) -> List[Reference]:
        """List incoming references to a document (backlinks)."""
        if not self._has_references_support():
            return []
        return self.backend.list_references_to(path)  # type: ignore[attr-defined]

    def all_references(self) -> List[Reference]:
        """List every reference in the store."""
        if not self._has_references_support():
            return []
        return self.backend.list_all_references()  # type: ignore[attr-defined]

    def walk_references(
        self,
        start_path: str,
        *,
        direction: str = "outgoing",
        max_depth: int = 2,
        ref_types: Optional[List[str]] = None,
        max_nodes: int = 50,
    ) -> Dict[str, Any]:
        """BFS walk through the reference graph from a starting document.

        Args:
            start_path: Document to start from.
            direction: 'outgoing' (follows from→to), 'incoming' (follows
                backlinks to→from), or 'both' (follows in either direction).
            max_depth: How many hops away from the start to traverse.
            ref_types: If given, only follow edges of these types.
            max_nodes: Cap on total visited nodes (truncates BFS).

        Returns:
            A dict with:
                - 'start': the starting path
                - 'edges': list of {from_path, to_path, ref_type, note, depth}
                - 'nodes': list of unique paths visited (with title + depth)
                - 'truncated': bool, True if max_nodes was hit
        """
        if not self._has_references_support():
            return {"start": start_path, "edges": [], "nodes": [], "truncated": False}

        type_filter = set(ref_types) if ref_types else None
        visited_paths: Dict[str, int] = {start_path: 0}  # path → depth first seen
        edges: List[Dict[str, Any]] = []
        truncated = False

        # BFS queue of (path, depth)
        queue: List[tuple] = [(start_path, 0)]
        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            outgoing: List[Reference] = []
            if direction in ("outgoing", "both"):
                outgoing.extend(self.references_from(current))
            incoming: List[Reference] = []
            if direction in ("incoming", "both"):
                incoming.extend(self.references_to(current))

            for r in outgoing:
                if type_filter and r.ref_type not in type_filter:
                    continue
                edges.append({
                    "from_path": r.from_path, "to_path": r.to_path,
                    "ref_type": r.ref_type, "note": r.note,
                    "depth": depth + 1,
                })
                if r.to_path not in visited_paths:
                    if len(visited_paths) >= max_nodes:
                        truncated = True
                        break
                    visited_paths[r.to_path] = depth + 1
                    queue.append((r.to_path, depth + 1))
            if truncated:
                break
            for r in incoming:
                if type_filter and r.ref_type not in type_filter:
                    continue
                edges.append({
                    "from_path": r.from_path, "to_path": r.to_path,
                    "ref_type": r.ref_type, "note": r.note,
                    "depth": depth + 1,
                    "incoming": True,
                })
                if r.from_path not in visited_paths:
                    if len(visited_paths) >= max_nodes:
                        truncated = True
                        break
                    visited_paths[r.from_path] = depth + 1
                    queue.append((r.from_path, depth + 1))
            if truncated:
                break

        # Collect node info with titles
        nodes: List[Dict[str, Any]] = []
        for path, d in visited_paths.items():
            doc = self.backend.read(path)
            nodes.append({
                "path": path,
                "title": doc.title if doc else "(missing)",
                "depth": d,
                "exists": doc is not None,
            })
        nodes.sort(key=lambda n: (n["depth"], n["path"]))
        return {
            "start": start_path, "edges": edges,
            "nodes": nodes, "truncated": truncated,
        }

    def render_reference_walk(
        self, walk: Dict[str, Any], indent: str = "  ",
    ) -> str:
        """Render a walk_references() result as an ASCII tree."""
        if not walk["edges"] and not walk["nodes"]:
            return f"No references from '{walk['start']}'."

        # Build adjacency from edges, grouped by depth via parent
        # Edge dict already includes from_path, to_path, ref_type, depth
        # We render as a tree rooted at start_path using outgoing direction.
        children: Dict[str, List[Dict[str, Any]]] = {}
        for e in walk["edges"]:
            # For incoming edges (from_path != current node), invert visually
            parent = e["from_path"]
            if e.get("incoming"):
                parent = e["to_path"]
                child = e["from_path"]
                arrow = "←"
            else:
                child = e["to_path"]
                arrow = "→"
            children.setdefault(parent, []).append({
                "child": child, "ref_type": e["ref_type"],
                "note": e.get("note", ""), "arrow": arrow,
            })

        node_titles = {n["path"]: n["title"] for n in walk["nodes"]}
        lines: List[str] = []
        seen_in_path: set = set()

        def render(path: str, depth: int) -> None:
            title = node_titles.get(path, "?")
            cycle = " [cycle]" if path in seen_in_path else ""
            lines.append(f"{indent * depth}{path} ({title}){cycle}")
            if path in seen_in_path:
                return
            seen_in_path.add(path)
            for edge in children.get(path, []):
                note = f" — {edge['note']}" if edge["note"] else ""
                lines.append(
                    f"{indent * (depth + 1)}{edge['arrow']}[{edge['ref_type']}]{note}"
                )
                render(edge["child"], depth + 2)
            seen_in_path.discard(path)

        render(walk["start"], 0)
        if walk["truncated"]:
            lines.append("... (truncated: max_nodes hit)")
        return "\n".join(lines)
