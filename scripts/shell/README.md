# Shell Scripts

This directory contains shell scripts for various setup, build, and deployment tasks.

## Scripts

### Environment Setup
- `fix_environment.sh` - Fix Python environment issues
- `init_vintageoptics.sh` - Initialize VintageOptics environment
- `setup_enhanced_gui.sh` - Setup enhanced GUI environment
- `setup_frontend.sh` - Setup frontend dependencies
- `setup_git_config.sh` - Configure git settings
- `setup_github_auth.sh` - Setup GitHub authentication

### Running the Application
- `run_enhanced_gui.sh` - Run with enhanced GUI
- `run_with_gui.sh` - Run with basic GUI
- `run_with_progress.sh` - Run with progress indicators
- `run_with_real_processing.sh` - Run with real processing enabled
- `run_with_separated_gui.sh` - Run with separated GUI components

### Git Operations
- `push_to_github.sh` - Push changes to GitHub
- `push_update.sh` - Push updates
- `push_vintageoptics_update.sh` - Push VintageOptics specific updates
- `quick_fix_push.sh` - Quick fix and push
- `check_repo_status.sh` - Check repository status

## Usage

Make scripts executable:
```bash
chmod +x scripts/shell/*.sh
```

Run a script:
```bash
./scripts/shell/init_vintageoptics.sh
```

## Note

These scripts may need to be updated to reflect the new directory structure. Always review a script before running it to ensure it's compatible with your environment.
