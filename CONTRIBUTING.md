# Contributing to NeuroAI Tutorials

Thank you for your interest in contributing to the NeuroAI Tutorials! This document provides guidelines for contributing to this educational project.

## 🎯 Types of Contributions

We welcome several types of contributions:

- **New tutorials** - Adding new tutorial notebooks on NeuroAI topics
- **Improvements** - Enhancing existing tutorials with better explanations or examples
- **Bug fixes** - Correcting errors in code or explanations
- **Documentation** - Improving setup instructions or adding clarifications

## 🚀 Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a new branch for your contribution:

   ```bash
   git checkout -b feature/your-tutorial-name
   ```

## 📝 Tutorial Guidelines

### Structure

Each tutorial should follow this structure:

1. **Title and Overview**

   - Clear, descriptive title
   - Brief overview of what will be covered
   - Learning objectives

2. **Prerequisites**

   - Required background knowledge
   - Previous tutorials (if applicable)

3. **Content Sections**

   - Explanatory markdown with theory and concepts
   - Code examples demonstrating concepts
   - Exercises for hands-on practice
   - Clear instructions for exercises

4. **Summary**
   - Key takeaways
   - References and further reading

### Code Standards

- Use Python 3.10+ features where appropriate
- Follow PEP 8 style guidelines (enforced by Ruff)
- Include type hints for function signatures
- Add docstrings for complex functions
- Keep cells focused and not too long
- Add comments explaining non-obvious code

### Markdown Guidelines

- Use clear, concise language
- Include diagrams or visualizations where helpful
- Break up long sections with subheadings
- Use code blocks with appropriate syntax highlighting
- Cite sources and provide references

## 🧪 Testing Your Tutorial

Before submitting:

1. **Run all cells** from start to finish to ensure no errors
2. **Clear outputs** if they're very large (but keep small, illustrative outputs)
3. **Test exercises** - Make sure they're solvable and have clear instructions
4. **Run linters**:

   ```bash
   ruff format tutorials/your-tutorial.ipynb
   nbqa ruff tutorials/your-tutorial.ipynb
   ```

## 📤 Submitting Your Contribution

1. **Commit your changes** with clear, descriptive commit messages:

   ```bash
   git add tutorials/your-tutorial.ipynb
   git commit -m "Add tutorial on [topic]"
   ```

2. **Push to your fork**:

   ```bash
   git push origin feature/your-tutorial-name
   ```

3. **Open a Pull Request**:
   - Provide a clear description of your tutorial
   - List the main topics covered
   - Mention any dependencies or prerequisites

## 📋 Pull Request Checklist

- [ ] Tutorial follows the structure guidelines
- [ ] All code cells run without errors
- [ ] Code follows Python style guidelines
- [ ] Markdown is clear and well-formatted
- [ ] Exercises have clear instructions
- [ ] References are cited where appropriate
- [ ] Commit messages are descriptive

## 🤝 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on educational value
- Help maintain a welcoming learning environment

## ❓ Questions?

If you have questions about contributing, please open an issue with the "question" label.

Thank you for helping make NeuroAI more accessible to learners! 🧠✨
