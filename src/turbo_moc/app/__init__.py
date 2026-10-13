"""TurboMoC browser interface; imported lazily for CLI and library consumers."""


def create_app(config_file=None, output_dir=None):
    """Create the app with an optional explicit workflow output directory."""
    from turbo_moc.app.application import create_app as factory
    return factory(config_file, output_dir=output_dir)


def main(argv=None):
    """Launch the canonical app without importing the preserved legacy UI."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Launch TurboMoC")
    parser.add_argument("--config", type=Path, help="Machine-project YAML to open")
    parser.add_argument("--port", type=int, default=8052)
    parser.add_argument("--output-dir", type=Path, help="Writable root for workflow outputs")
    args = parser.parse_args(argv)
    create_app(args.config, output_dir=args.output_dir).run(
        host="127.0.0.1", port=args.port, debug=False)
