# CLAUDE.md - AI Assistant Guide for cfb Repository

## Project Overview

**cfb** (College Football 2025: Early Experimentation) is a data analytics project for college football betting strategy. The project focuses on:

- Analyzing SP+ (Bill Connelly's college football prediction metric) data
- Monitoring Kalshi betting platform positions
- Building a decision framework for sports betting selections
- Tracking wagers and outcomes

This is an early-stage experimental/research project using Jupyter notebooks for exploratory data analysis.

## Repository Structure

```
cfb/
├── CLAUDE.md                    # This file - AI assistant guide
├── README.md                    # Project overview and weekly workflow
├── requirements.txt             # Python dependencies (113 packages)
├── .gitignore                   # Git exclusions (data/, venv, etc.)
├── b12_schedule_analysis.ipynb  # Main analysis notebook
└── data/                        # (gitignored) SP+ Excel files
```

## Tech Stack

- **Python**: 3.11.11
- **Environment**: Virtual environment (.env directory)
- **Primary Interface**: JupyterLab for interactive analysis
- **Core Libraries**:
  - `pandas` (2.3.2) - Data manipulation
  - `numpy` (2.3.2) - Numerical operations
  - `matplotlib` (3.10.5) - Visualization
  - `openpyxl` (3.1.5) - Excel file I/O
  - `beautifulsoup4` (4.13.5) - Web scraping
  - `requests` (2.32.5) / `httpx` (0.28.1) - HTTP clients

## Development Setup

```bash
# Create and activate virtual environment
python -m venv .env
source .env/bin/activate  # On Unix/macOS

# Install dependencies
pip install -r requirements.txt

# Launch JupyterLab
jupyter lab
```

## Data Conventions

### File Naming
- SP+ data files: `data/sp+_YYYY-MM-DD.xlsx` (e.g., `data/sp+_2025-08-13.xlsx`)
- Date format is ISO 8601 (YYYY-MM-DD)

### Data Structure
- SP+ Excel files contain team rankings with format: `{rank}. {Team Name}`
- Example: "1. Ohio St.", "2. Alabama", etc.
- Currently tracking 136 college football teams

### Important: Data Directory
- The `data/` directory is **gitignored** (contains sensitive betting/research data)
- Never commit data files to the repository
- Data files must be obtained separately

## Weekly Workflow

The project follows this weekly cycle (from README.md):

1. **Update SP+ Data** - Fetch latest rankings
2. **Scan Kalshi Positions** - Review betting platform
   - SWOT analysis for each position
   - Review/confirm current orders
3. **Narrow Selection** - Apply "JDK Adjustment" filtering
4. **Record Best Bets** - Document top picks
5. **Budget Units** - Allocate betting units
6. **Place Splash Selections** - Execute bets
7. **Record All Wagers/Outcomes** - Track results

## Code Patterns

### Data Loading Pattern
```python
import pandas as pd

date = 'YYYY-MM-DD'
cur_sp_plus_df = pd.read_excel(f'data/sp+_{date}.xlsx')
```

### Data Cleaning Pattern
Team names need to be split from ranking numbers:
```python
# Raw format: "1. Ohio St."
cur_sp_plus_df['TEAM'].str.split(' ')
# Results in: ["1.", "Ohio", "St."]
```

## Key Guidelines for AI Assistants

### Do
- Use pandas for all data manipulation
- Follow existing date-based file naming conventions
- Keep analysis in Jupyter notebooks for this exploratory phase
- Preserve the weekly workflow structure
- Use openpyxl for Excel file operations

### Don't
- Commit anything to the `data/` directory
- Hardcode dates - use variables for date parameters
- Add production-style code structure (project is intentionally exploratory)
- Modify the weekly workflow without explicit user approval

### External References
- Detailed planning docs: [Google Doc](https://docs.google.com/document/d/1BKvo23DyGppxvRifqFgwkUoflPLQX5wabjUDIV4qUZA/edit?tab=t.0)
- SP+ metric documentation: Bill Connelly's college football analytics

## Git Workflow

- **Main branch**: Development happens on feature branches
- **Commits**: Use descriptive commit messages
- **Data exclusion**: Ensure `data/` never gets committed

## Common Tasks

### Adding New Analysis
1. Create cells in existing notebook or new `.ipynb` file
2. Follow the data loading pattern with date variable
3. Document findings with markdown cells

### Updating Dependencies
```bash
pip freeze > requirements.txt
```

### Running the Notebook
```bash
jupyter lab b12_schedule_analysis.ipynb
```

## Project Status

This is an **early-stage experimental project**. The codebase is intentionally minimal:
- Single Jupyter notebook for analysis
- No src/tests/docs directory structure
- Focus on rapid iteration and exploration
- Production patterns should not be introduced prematurely
