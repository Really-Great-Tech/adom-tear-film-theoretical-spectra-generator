#!/usr/bin/env python3
"""
Cross-platform runner for the tear film theoretical spectra generator.
This script handles path setup and runs the generator with default configurations.
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse
import yaml

def main():
    parser = argparse.ArgumentParser(description="Run Tear Film generator or UI")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit UI instead of batch generator")
    args, unknown = parser.parse_known_args()
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Change to project root directory
    os.chdir(project_root)
    
    # Decide mode (auto-enable UI if config.yaml sets ui.enabled: true)
    ui_enabled_config = False
    try:
        with open(project_root / "config.yaml", "r") as f:
            cfg = yaml.safe_load(f) or {}
            ui_enabled_config = bool(cfg.get("ui", {}).get("enabled", False))
    except Exception:
        pass

    launch_ui = args.ui or ui_enabled_config

    if launch_ui:
        streamlit_app = project_root / "src" / "streamlit_app.py"
        if not streamlit_app.exists():
            print(f"Error: Streamlit app not found at {streamlit_app}")
            sys.exit(1)
        cmd = [sys.executable, "-m", "streamlit", "run", str(streamlit_app)]
        # pass through unknown args to streamlit if any
        if unknown:
            cmd.extend(unknown)
    else:
        # Default configuration for sample generation
        python_script = project_root / "src" / "tear_film_generator.py"
        if not python_script.exists():
            print(f"Error: Python script not found at {python_script}")
            sys.exit(1)
        cmd = [sys.executable, str(python_script)]
    
    print("=== Running Tear Film Theoretical Spectra Generator ===")
    print(f"Working directory: {project_root}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run the generator or UI
        result = subprocess.run(cmd, check=True)
        if not launch_ui:
            print("\n=== Generation completed successfully! ===")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError: Generator failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
