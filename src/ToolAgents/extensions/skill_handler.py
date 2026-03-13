# src/ToolAgents/extensions/skill_handler.py
"""SkillFolderHandler — implements the Agent Skills specification for SKILL.md files."""
from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import yaml

from ToolAgents.function_tool import FunctionTool

from .models import ActivationResult, ExtensionEntry

if TYPE_CHECKING:
    from .manager import ExtensionManager

logger = logging.getLogger(__name__)

_RESOURCE_DIRS = ("scripts", "references", "assets")
_MAX_RESOURCES = 50


def _parse_frontmatter(text: str) -> tuple[Optional[dict], str]:
    if not text.startswith("---"):
        return None, text
    end = text.find("---", 3)
    if end == -1:
        return None, text
    yaml_str = text[3:end].strip()
    body = text[end + 3:].strip()
    try:
        data = yaml.safe_load(yaml_str)
    except yaml.YAMLError:
        try:
            lines = []
            for line in yaml_str.split("\n"):
                if ":" in line and not line.strip().startswith("#"):
                    key, _, val = line.partition(":")
                    val = val.strip()
                    if val and not val.startswith(("'", '"', "[", "{", "|", ">")):
                        line = f'{key}: "{val}"'
                lines.append(line)
            data = yaml.safe_load("\n".join(lines))
        except yaml.YAMLError:
            return None, body
    if not isinstance(data, dict):
        return None, body
    return data, body


class SkillFolderHandler:
    name: str = "skills"
    marker_file: str = "SKILL.md"

    def discover(self, path: Path) -> Optional[ExtensionEntry]:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.error("Failed to read %s: %s", path, e)
            return None
        frontmatter, _ = _parse_frontmatter(text)
        if frontmatter is None:
            logger.error("Unparseable YAML in %s", path)
            return None
        name = frontmatter.get("name", "")
        description = frontmatter.get("description", "")
        if not description:
            logger.error("Missing description in %s — skipping", path)
            return None
        if not name:
            name = path.parent.name
        dir_name = path.parent.name
        if name != dir_name:
            logger.warning("Skill name '%s' does not match directory '%s' in %s", name, dir_name, path)
        if len(name) > 64:
            logger.warning("Skill name '%s' exceeds 64 characters in %s", name, path)
        metadata = {}
        for key in ("license", "compatibility", "allowed-tools"):
            if key in frontmatter:
                metadata[key] = frontmatter[key]
        if "metadata" in frontmatter and isinstance(frontmatter["metadata"], dict):
            metadata.update(frontmatter["metadata"])
        pin_value = frontmatter.get("pin_in_context")
        if pin_value is not None:
            metadata["pin_in_context"] = bool(pin_value)
        return ExtensionEntry(name=name, description=description, handler_type=self.name,
                            path=path, scope="", priority=0, metadata=metadata)

    def build_catalog(self, entries: List[ExtensionEntry]) -> str:
        if not entries:
            return ""
        lines = [
            "The following skills provide specialized instructions for specific tasks.",
            "When a task matches a skill's description, call the activate_skill tool",
            "with the skill's name to load its full instructions.",
            "Users can also activate skills directly with /skill-name commands.",
            "",
            "<available_skills>",
        ]
        for entry in sorted(entries, key=lambda e: e.name):
            lines.append(f'  <skill name="{entry.name}" location="{entry.path}">')
            lines.append(f"    {entry.description}")
            lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)

    def activate(self, entry: ExtensionEntry) -> ActivationResult:
        text = entry.path.read_text(encoding="utf-8")
        _, body = _parse_frontmatter(text)
        skill_dir = entry.path.parent
        resources = self._enumerate_resources(skill_dir)
        content_lines = [f'<skill_content name="{entry.name}">']
        content_lines.append(body)
        content_lines.append("")
        content_lines.append(f"Skill directory: {skill_dir}")
        content_lines.append("Relative paths in this skill resolve against the skill directory.")
        if resources:
            content_lines.append("")
            content_lines.append("<skill_resources>")
            for res in resources:
                content_lines.append(f"  <file>{res}</file>")
            content_lines.append("</skill_resources>")
        content_lines.append("</skill_content>")
        pin = entry.metadata.get("pin_in_context", True)
        return ActivationResult(content="\n".join(content_lines), pin_in_context=pin,
                               resources=resources if resources else None)

    def get_tools(self, manager: "ExtensionManager") -> List[FunctionTool]:
        skill_entries = [e for e in manager.entries.values() if e.handler_type == self.name]
        if not skill_entries:
            return []
        skill_names = sorted(e.name for e in skill_entries)
        SkillNameEnum = Enum("SkillNameEnum", {n: n for n in skill_names})

        def activate_skill(skill_name: SkillNameEnum) -> str:
            """Activate a skill by name to load its full instructions into context.

            Call this when a task matches one of the available skills.

            Args:
                skill_name: Name of the skill to activate.

            Returns:
                The skill's full instructions wrapped in structured tags.
            """
            name_str = skill_name.value if isinstance(skill_name, Enum) else str(skill_name)
            result = manager.activate(name_str)
            if result is None:
                return f"Error: skill '{name_str}' not found."
            manager._pending_activations[name_str] = result
            return result.content

        tool = FunctionTool(activate_skill)
        tool.model.__name__ = "activate_skill"
        # Rebuild the model so Pydantic can resolve the locally-defined SkillNameEnum
        tool.model.model_rebuild(_types_namespace={"SkillNameEnum": SkillNameEnum})
        return [tool]

    @staticmethod
    def _enumerate_resources(skill_dir: Path) -> List[str]:
        resources = []
        for subdir_name in _RESOURCE_DIRS:
            subdir = skill_dir / subdir_name
            if not subdir.is_dir():
                continue
            for file_path in sorted(subdir.rglob("*")):
                if file_path.is_file():
                    rel = file_path.relative_to(skill_dir)
                    resources.append(str(rel).replace("\\", "/"))
        resources.sort()
        if len(resources) > _MAX_RESOURCES:
            logger.warning("Skill '%s' has %d resource files, truncating to %d",
                          skill_dir.name, len(resources), _MAX_RESOURCES)
            resources = resources[:_MAX_RESOURCES]
        return resources
