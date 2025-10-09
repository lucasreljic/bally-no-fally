# bally-no-fally

MTE 380 Project to keep a pingpong ball centered on a table.

## Development Setup

Create a virtual environment and source it.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install deps through requirements.txt

```bash
pip install -r requirements.txt
```

This project uses pre-commit hooks to ensure code quality and consistency.

### Installing Pre-commit

Pre-commit is already installed in the virtual environment. To set up the git hooks:

```bash
pre-commit install
```

### Running Pre-commit

To run all pre-commit hooks on all files:

```bash
pre-commit run --all-files
```

To run pre-commit on specific files:

```bash
pre-commit run --files path/to/file.py
```

### Pre-commit Hooks Included

- **Code Formatting**: Black (Python code formatter)
- **Import Sorting**: isort (Python import organizer)
- **Linting**: Ruff (Fast Python linter)
- **Type Checking**: mypy (Static type checker)
- **Security**: Bandit (Security linter)
- **Documentation**: pydocstyle (Docstring style checker)
- **Secrets Detection**: detect-secrets (Prevents secrets in code)
- **General**: Various file formatting and cleanup hooks
- **Markdown**: markdownlint (Markdown formatting)

Pre-commit hooks will run automatically on every commit to ensure code quality.

## Systemd

The code is intended to start automatically on the Raspberry Pi using systemd and the  `bally-no-fally.service` file

### Setup

First copy over the `bally-no-fally.service` file to `/etc/systemd/system/`

Then run the following commands

```bash
sudo systemctl daemon-reload
sudo systemctl enable bally-no-fally.service
```

**That's it!** Now it will run automatically on boot!

## VNC

In order to see the Raspberry Pi desktop you must ssh using this command

```bash
sudo ssh -L 5900:localhost:5900 lucas@raspberrypi.local
```

Then use a vnc client (Only been tested with RealVNC).
Connect with the address `localhost:5900`, that's all!
