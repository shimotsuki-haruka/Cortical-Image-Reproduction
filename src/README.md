# FluoroMind Source Directory

This directory contains the main source code for the FluoroMind package, a comprehensive platform for fluorescence microscopy data analysis.

## Package Structure

```
fluoromind/
├── __init__.py          # Package initialization
├── algorithms/          # Advanced analysis algorithms
│   ├── caps.py         # Calcium activity pattern similarity
│   ├── clustering.py   # Clustering algorithms
│   ├── correlation.py  # Correlation analysis
│   ├── cpca.py        # Contrastive PCA
│   ├── fc.py          # Functional connectivity
│   ├── parallel.py    # Parallel processing utilities
│   └── swc.py         # SWC file processing
├── core/               # Core functionality
│   ├── analysis.py    # Basic analysis tools
│   ├── group.py       # Group analysis
│   ├── pipeline.py    # Analysis pipeline framework
│   ├── preprocessing.py# Data preprocessing
│   └── stats.py       # Statistical utilities
├── io/                 # Input/Output operations
│   ├── handlers.py    # File handlers
│   ├── readers.py     # Data readers
│   └── storage.py     # Data storage utilities
├── py.typed           # Type hint marker
└── server/            # HTTP server implementation
    ├── __main__.py    # Server entry point
    ├── app.py         # FastAPI application
    ├── cli.py         # Command-line interface
    └── routes.py      # API endpoints
```

## Components

### Core Module
- **analysis**: Basic analysis tools and utilities
- **preprocessing**: Data preprocessing and image processing functions
- **group**: Group-level analysis tools
- **pipeline**: Framework for creating analysis pipelines
- **stats**: Statistical analysis utilities

### Algorithms Module
- **caps**: Calcium activity pattern similarity analysis
- **clustering**: Advanced clustering algorithms
- **correlation**: Correlation analysis methods
- **cpca**: Contrastive Principal Component Analysis
- **fc**: Functional connectivity analysis
- **parallel**: Parallel processing utilities
- **swc**: SWC file format processing tools

### IO Module
- **handlers**: File format handlers
- **readers**: Data import utilities
- **storage**: Data storage and management

### Server Module
The server provides HTTP access to FluoroMind's functionality. To use it:

```bash
# Install server dependencies
pip install fluoromind[server]

# Start the server
python -m fluoromind.server start

# Show available options
python -m fluoromind.server --help
```

Server options:
- `--host`: Bind address (default: 127.0.0.1)
- `--port`: Port number (default: 8000)
- `--reload`: Enable auto-reload for development
- `--workers`: Number of worker processes
- `--log-level`: Logging level

## Development

When adding new features:
1. Place core functionality in appropriate modules under `core/`
2. Add specialized algorithms in `algorithms/`
3. Put I/O related code in `io/`
4. Add HTTP endpoints in `server/routes.py` if needed
5. Update `__init__.py` to expose new functionality
6. Include proper type hints and documentation (package is typed)
7. Add tests in the `tests/` directory (parallel to `src/`)
