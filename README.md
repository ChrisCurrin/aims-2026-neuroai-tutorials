# NeuroAI Tutorials

Welcome to the NeuroAI Tutorials repository! This collection of interactive Jupyter notebooks provides hands-on learning materials for exploring the intersection of neuroscience and artificial intelligence.

## 🎯 Overview

These tutorials are designed for students taking the NeuroAI course. Each tutorial combines:

- **Informative markdown content** that explains key concepts
- **Interactive code exercises** for hands-on problem-solving
- **Modern Python standards** using the latest best practices

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) - A fast Python package installer and resolver

### Installation

1. **Install uv** (if not already installed):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:

   ```bash
   git clone https://github.com/ChrisCurrin/neuroai-tutorials.git
   cd neuroai-tutorials
   ```

3. **Create a virtual environment and install dependencies**:

   ```bash
   uv sync --all-extras
   ```

4. **Launch Jupyter**:

   ```bash
   jupyter notebook
   ```

## 📚 Tutorial Structure

Tutorials are organized by topic in the `tutorials/` directory:

```
tutorials/
├── 00_the_neuron_from_scratch/ # <-- background material
│   └── ...
├── 11_introduction/ # <-- Week 1 day 1
│   └── what_is_neuroai.ipynb
├── 12_the_neuron/ # <-- Week 1 day 2
│   └── neuron_and_perceptron.ipynb
└── ...
```

Each tutorial notebook contains:

- **Learning objectives** - What you'll learn
- **Concept explanations** - Theory and background
- **Code examples** - Demonstrations of key concepts
- **Exercises** - Hands-on problems to solve
- **Solutions** - Reference implementations (in separate files)

## 🛠️ Development

### Code Quality

We use modern Python tooling to maintain code quality:

- **Ruff** - Fast Python linter and formatter
- **pre-commit** - Git hooks for automated checks

To set up pre-commit hooks:

```bash
pre-commit install
```

### Running Linters

Format and lint notebooks:

```bash
ruff format .
ruff check .
nbqa ruff tutorials/
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋 Support

For questions or issues, please open an issue on GitHub.

---

Happy Learning! 🧠✨
