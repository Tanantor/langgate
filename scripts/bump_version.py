#!/usr/bin/env python3
"""
Version bumping script for LangGate.

This script automatically updates version numbers across the project:
- Main pyproject.toml
- Package pyproject.toml files
- Helm chart Chart.yaml files

Usage:
    python scripts/bump_version.py [major|minor|patch|--version X.Y.Z]
"""

import argparse
import logging
import re
import sys
import tomllib
from pathlib import Path
from typing import Literal

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

VersionType = Literal["major", "minor", "patch"]


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse a semantic version string into a tuple of (major, minor, patch)."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version)
    if not match:
        raise ValueError(f"Invalid version format: {version}")
    major, minor, patch = map(int, match.groups())
    return (major, minor, patch)


def bump_version(current_version: str, bump_type: VersionType) -> str:
    """Bump the version according to semantic versioning."""
    major, minor, patch = parse_version(current_version)

    if bump_type == "major":
        return f"{major + 1}.0.0"
    if bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    if bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"Invalid bump type: {bump_type}")


def find_pyproject_files(project_root: Path) -> list[Path]:
    """Find all pyproject.toml files in the project."""
    result = [project_root / "pyproject.toml"]

    # Find package pyproject files
    packages_dir = project_root / "packages"
    if packages_dir.exists():
        for package_dir in packages_dir.iterdir():
            if package_dir.is_dir():
                pyproject = package_dir / "pyproject.toml"
                if pyproject.exists():
                    result.append(pyproject)

    return result


def find_chart_files(project_root: Path) -> list[Path]:
    """Find all Chart.yaml files in the project."""
    result = []

    charts_dir = project_root / "deployment" / "k8s" / "charts"
    if charts_dir.exists():
        # Check top-level charts
        for chart_dir in charts_dir.iterdir():
            if chart_dir.is_dir():
                chart_yaml = chart_dir / "Chart.yaml"
                if chart_yaml.exists():
                    result.append(chart_yaml)

        # Check library charts
        library_dir = charts_dir / "library"
        if library_dir.exists() and library_dir.is_dir():
            for lib_chart_dir in library_dir.iterdir():
                if lib_chart_dir.is_dir():
                    chart_yaml = lib_chart_dir / "Chart.yaml"
                    if chart_yaml.exists():
                        result.append(chart_yaml)

    return result


def update_pyproject_toml(file_path: Path, new_version: str) -> None:
    """Update version in a pyproject.toml file."""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    pattern = r'(\[project\][^\[]*?\bversion\s*=\s*")[^"]*(")'
    # Use named groups to avoid numeric reference issues
    replacement = r"\g<1>" + new_version + r"\g<2>"
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    if new_content == content:
        logger.error(f"Could not find project.version in {file_path}")
        return

    # Update langgate package version constraints in dependencies and optional dependencies sections
    # This will find and update langgate-* package versions in (==X.Y.Z) format
    new_content = re.sub(
        r'("langgate-[^"]+)(\s*\(==)[^)]*(\)")',
        r"\g<1>\g<2>" + new_version + r"\g<3>",
        new_content,
    )
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    logger.info(f"Updated {file_path} to version {new_version}")


def update_chart_yaml(file_path: Path, new_version: str) -> None:
    """Update version in a Chart.yaml file using direct text replacement."""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Update version field - use lambda to avoid backreference issues
    content = re.sub(
        pattern=r"(version:\s*)[\d\.]+",
        repl=lambda m: f"{m.group(1)}{new_version}",
        string=content,
    )

    # Update appVersion field - ensure it's quoted
    content = re.sub(
        pattern=r"(appVersion:\s*)[\"\']?[\d\.]+[\"\']?",
        repl=lambda m: f'{m.group(1)}"{new_version}"',
        string=content,
    )

    # Update dependency versions for langgate components
    content = re.sub(
        pattern=r"(- name: langgate[^\n]+\n\s+repository:[^\n]+\n\s+version:\s*)[\d\.]+",
        repl=lambda m: f"{m.group(1)}{new_version}",
        string=content,
    )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Updated {file_path} to version {new_version}")


def find_current_version() -> str:
    """Find the current version from the main pyproject.toml file."""
    main_pyproject = Path(__file__).parent.parent / "pyproject.toml"
    if not main_pyproject.exists():
        raise FileNotFoundError(f"Could not find {main_pyproject}")

    with open(main_pyproject, "rb") as f:
        data = tomllib.load(f)

    if "project" in data and "version" in data["project"]:
        return data["project"]["version"]

    raise ValueError("Could not find version in main pyproject.toml")


def update_pyproject_versions(project_root: Path, new_version: str) -> None:
    """Update version in all pyproject.toml files."""
    for pyproject_file in find_pyproject_files(project_root):
        update_pyproject_toml(pyproject_file, new_version)


def update_chart_versions(project_root: Path, new_version: str) -> None:
    """Update version in all Chart.yaml files."""
    for chart_file in find_chart_files(project_root):
        update_chart_yaml(chart_file, new_version)


def update_versions(new_version: str) -> None:
    """Update versions across the project."""
    project_root = Path(__file__).parent.parent
    update_pyproject_versions(project_root, new_version)
    update_chart_versions(project_root, new_version)


def validate_pyproject_versions(project_root: Path, expected_version: str) -> list[str]:
    """Validate versions in pyproject.toml files, returns inconsistencies."""
    inconsistencies = []

    for pyproject_file in find_pyproject_files(project_root):
        with open(pyproject_file, "rb") as f:
            data = tomllib.load(f)

        if (
            "project" in data
            and "version" in data["project"]
            and data["project"]["version"] != expected_version
        ):
            inconsistencies.append(
                f"{pyproject_file}: {data['project']['version']} != {expected_version}"
            )

    return inconsistencies


def validate_chart_versions(project_root: Path, expected_version: str) -> list[str]:
    """Validate versions in Chart.yaml files, returns inconsistencies."""
    inconsistencies = []

    for chart_file in find_chart_files(project_root):
        with open(chart_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "version" in data and data["version"] != expected_version:
            inconsistencies.append(
                f"{chart_file}: {data['version']} != {expected_version}"
            )

    return inconsistencies


def validate_versions() -> bool:
    """Validate that all versions are consistent."""
    project_root = Path(__file__).parent.parent
    main_version = find_current_version()

    inconsistencies = []
    inconsistencies.extend(validate_pyproject_versions(project_root, main_version))
    inconsistencies.extend(validate_chart_versions(project_root, main_version))

    if inconsistencies:
        logger.error("Version inconsistencies found:")
        for inconsistency in inconsistencies:
            logger.error(f"  - {inconsistency}")
        return False

    logger.info("All versions are consistent!")
    return True


def main() -> None:
    """Run the version bumping script."""
    parser = argparse.ArgumentParser(
        description="Bump version across all LangGate components"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "bump_type",
        nargs="?",
        choices=["major", "minor", "patch"],
        help="Type of version bump to perform",
    )
    group.add_argument(
        "--version",
        help="Explicitly set version to this value (format: X.Y.Z)",
    )
    group.add_argument(
        "--validate",
        action="store_true",
        help="Validate that all versions are consistent",
    )

    args = parser.parse_args()

    if args.validate:
        sys.exit(0 if validate_versions() else 1)

    current_version = find_current_version()
    new_version = ""

    if args.version:
        # Validate the provided version
        try:
            parse_version(args.version)
            new_version = args.version
        except ValueError as e:
            logger.error(f"Error: {e}")
            sys.exit(1)
    else:
        new_version = bump_version(current_version, args.bump_type)

    logger.info(f"Bumping version from {current_version} to {new_version}")
    update_versions(new_version)
    logger.info(f"\nSuccessfully updated all versions to {new_version}")
    logger.info(
        "\nDon't forget to update CHANGELOG.md with a summary of changes"
        " before creating a release."
    )


if __name__ == "__main__":
    main()
