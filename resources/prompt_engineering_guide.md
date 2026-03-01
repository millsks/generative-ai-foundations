# Prompt Engineering for Software Development & Generative AI
*A Practical, Example-Driven Guide*

## 1. What Is Prompt Engineering (for Devs)?
Prompt engineering is the discipline of designing inputs to LLMs to reliably get useful, high-quality outputs. For developers, this means turning vague ideas into precise, reproducible prompts that can collaborate with codebases and APIs.

### 1.1 Core Principles
1. **Explicitness over Implicitness**: Tell the model exactly what you want.
2. **Decomposition over Monoliths**: Break complex tasks into smaller steps.
3. **Iteration over Perfection**: Start basic, then refine.
4. **Grounding in Context**: Provide necessary code, data, and constraints.

---

## 2. Prompt Anatomy: The "SCQR-F" Pattern
- **S – System / Role**: Who the model should act as.
- **C – Context**: Background, domain info, code, files.
- **Q – Question / Task**: What you want.
- **R – Response Format**: How to respond (markdown, JSON, etc.).
- **F – Feedback Loop**: How you’ll iterate.

### 2.1 Example: Refactoring Legacy Code
```markdown
You are an expert software engineer with 10+ years of experience in refactoring Python codebases.

Context:
- Language: Python 3.11
- Framework: FastAPI
- The code mixes I/O and business logic.

Task:
Given the code snippet below:
[INSERT CODE HERE]

1. Identify the main design smells.
2. Propose a refactored structure.
3. Provide the refactored code with comments.

Response format:
- Use markdown.
- Sections: "### Smells Found", "### Proposed Design", "### Refactored Code".
```

---

## 3. Foundational Prompt Patterns

### 3.1 Role Prompting
```markdown
You are a senior backend engineer specializing in PostgreSQL. 
Review the following schema for performance and indexing strategy.
```

### 3.2 Few-Shot Prompting (Examples)
```markdown
You generate unit tests for Python using pytest.

Input:
def add(a, b): return a + b

Output:
import pytest
def test_add(): assert add(2, 3) == 5

Now generate tests for:
def multiply(a, b): return a * b
```

---

## 4. Essential Patterns for Dev Tasks

### 4.1 Code Generation
```markdown
You are a senior engineer. Generate a FastAPI project skeleton that:
- Has a /health endpoint.
- Uses Pydantic for validation.
- Includes a Dockerfile.

Format: Tree structure followed by file contents.
```

### 4.2 Bug Hunting
```markdown
You are a debugging assistant. Find probable bugs in this Javascript middleware.
[INSERT CODE]
```

---

## 5. Iteration & Debugging
If the output is wrong:
1. **Too vague?** Add constraints.
2. **Hallucinations?** Tell it "If unsure, say I don't know."
3. **Wrong format?** Provide a strict schema.

---

## 6. Checklist: Is This a Good Prompt?
- [ ] Role is clear.
- [ ] Context is sufficient.
- [ ] Task is specific.
- [ ] Constraints are explicit.
- [ ] Output format is defined.
