#!/usr/bin/env python3
"""
Envoy configuration template renderer.
Renders Jinja2 templates with environment variables and outputs the result to a file.
"""

import argparse
import json
import os
import sys
from typing import Any

DEBUG = os.environ.get("DEBUG") == "1"

try:
    import jinja2
except ImportError:
    sys.stderr.write(
        'ERROR: Jinja2 library not found. Please install it with "pip install jinja2"\n'
    )
    sys.exit(1)


class TemplateVariableProvider:
    """Base class for providing template variables from different sources."""

    def get_variables(self) -> dict[str, Any]:
        """Get variables from the source."""
        raise NotImplementedError("Subclasses must implement get_variables")


class EnvironmentVariableProvider(TemplateVariableProvider):
    """Provides template variables from environment variables."""

    def __init__(self) -> None:
        """Initialize the provider."""

    @staticmethod
    def clean_value(value: str) -> str:
        """Clean environment variable values by removing unnecessary quotes."""
        if isinstance(value, str) and (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            return value[1:-1]
        return value

    def get_variables(self) -> dict[str, Any]:
        """Get all environment variables as template variables."""
        variables = {}
        for key, value in os.environ.items():
            # Convert to lowercase for template use
            template_key = key.lower()
            clean_value = self.clean_value(value)
            variables[template_key] = clean_value
        return variables


class FileVariableProvider(TemplateVariableProvider):
    """Provides template variables from a JSON file."""

    def __init__(self, file_path: str) -> None:
        """Initialize with the path to the JSON file."""
        self.file_path = file_path

    def get_variables(self) -> dict[str, Any]:
        """Load variables from the JSON file."""
        try:
            with open(self.file_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            sys.stderr.write(
                f"ERROR: Failed to load variables from {self.file_path}: {e}\n"
            )
            sys.exit(1)


class TemplateRenderer:
    """Renders templates using Jinja2."""

    def __init__(self, template_file: str, output_file: str) -> None:
        """Initialize with template and output file paths."""
        self.template_file = template_file
        self.output_file = output_file
        self.variables: dict[str, Any] = {}

    def add_variables(self, variables: dict[str, Any]) -> None:
        """Add variables to be used in template rendering."""
        self.variables.update(variables)

    def set_default(self, key: str, value: Any) -> None:
        """Set a default value for a variable if it doesn't exist."""
        if key not in self.variables:
            self.variables[key] = value

    def render(self) -> None:
        """Render the template with the provided variables."""
        try:
            # Load the template
            with open(self.template_file) as f:
                template_content = f.read()

            template = jinja2.Template(template_content)
            rendered_content = template.render(**self.variables)

            with open(self.output_file, "w") as f:
                f.write(rendered_content)

            # Save a copy with the debug suffix if in debug mode
            if DEBUG:
                debug_output_file = f"{self.output_file}.debug"
                with open(debug_output_file, "w") as f:
                    f.write(rendered_content)
                print(f"Debug copy saved to {debug_output_file}")

            print(f"Successfully rendered {self.template_file} to {self.output_file}")

        except Exception as e:
            sys.stderr.write(f"ERROR: Failed to render template: {e}\n")
            sys.exit(1)


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Render Envoy configuration templates using environment variables."
    )
    parser.add_argument(
        "--template", "-t", required=True, help="Path to the template file"
    )
    parser.add_argument("--output", "-o", required=True, help="Path to the output file")
    parser.add_argument(
        "--vars", "-v", help="Path to JSON file with additional template variables"
    )

    args = parser.parse_args()

    renderer = TemplateRenderer(args.template, args.output)

    env_provider = EnvironmentVariableProvider()
    renderer.add_variables(env_provider.get_variables())

    # Add variables from file if provided (overrides environment vars)
    if args.vars:
        file_provider = FileVariableProvider(args.vars)
        renderer.add_variables(file_provider.get_variables())

    # Set default values for required variables
    renderer.set_default("langgate_host", "langgate")

    print(f"Rendering template {args.template} with variables:")
    for key, value in sorted(renderer.variables.items()):
        if "api_key" in key.lower() or "secret" in key.lower():
            print(f"  {key}: [REDACTED]")
        else:
            print(f"  {key}: {value}")

    if DEBUG:
        print("\nRAW ENVIRONMENT VARIABLES:")
        for key, value in sorted(os.environ.items()):
            if "api_key" in key.lower() or "secret" in key.lower():
                print(f"  {key}: [REDACTED]")
            else:
                print(f"  {key}: {value}")

    renderer.render()


if __name__ == "__main__":
    main()
