"""Prompt bank for verbatim-core extraction and template generation.

Uses Jinja2 templates for variable substitution and conditionals.
Prompt files use standard Jinja2 syntax:
  - Variables: {{ variable_name }}
  - Conditionals: {% if condition %}...{% endif %}
  - Else: {% if condition %}...{% else %}...{% endif %}
"""

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent

_env = Environment(
    loader=FileSystemLoader(str(PROMPTS_DIR)),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_prompt(prompt_template: str, **kwargs) -> str:
    """Render a prompt template string with Jinja2.

    :param prompt_template: Jinja2 template string
    :param kwargs: template variables
    :return: rendered prompt string
    """
    template = _env.from_string(prompt_template)
    return template.render(**kwargs)


def load_prompt(name: str, **kwargs) -> str:
    """Load and optionally render a prompt template from the prompt bank.

    :param name: Prompt name (e.g. 'extraction/default')
    :param kwargs: if provided, render the template with these variables
    :return: Prompt template string (raw if no kwargs, rendered if kwargs given)
    """
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        msg = f"Prompt not found: {name} (looked in {path})"
        raise FileNotFoundError(msg)
    if kwargs:
        template = _env.get_template(f"{name}.txt")
        return template.render(**kwargs)
    return path.read_text(encoding="utf-8")


def list_prompts() -> list[str]:
    """List all available prompt templates.

    :return: List of prompt names
    """
    prompts = []
    for p in PROMPTS_DIR.rglob("*.txt"):
        name = str(p.relative_to(PROMPTS_DIR)).removesuffix(".txt")
        prompts.append(name)
    return sorted(prompts)
