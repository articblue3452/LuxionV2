<p align="center">
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&height=250&color=gradient&customColorList=12,19,24,30&text=Luxion%20V2&fontSize=55&fontColor=ffffff&animation=fadeIn&fontAlignY=38"/>
</p>

<p align="center">
<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=26&pause=1200&color=00F5FF&center=true&vCenter=true&width=900&lines=Goal-Driven+AI+Agent;Memory+%7C+Planning+%7C+Reflection;LangGraph+Powered;Ollama+Compatible;Autonomous+Tool+Execution;Built+by+ArticBlue"/>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent-green?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

</p>

---

# 🤖 Luxion V2

> **A Goal-Driven AI Agent with Memory, Planning, Reflection, Progress Tracking and Tool Execution.**

Luxion V2 is an experimental autonomous AI agent built using **LangGraph**.

Instead of simply answering prompts, Luxion reasons through a complete agent workflow:

- Understand the user's goal
- Retrieve relevant memories
- Build contextual understanding
- Plan actions
- Execute tools
- Track progress
- Reflect on outcomes
- Store new memories

---

# ✨ Features

✅ Goal Understanding

✅ Long-Term Memory

✅ Context Building

✅ Dynamic Planning

✅ Tool Execution

✅ Reflection Loop

✅ Progress Tracking

✅ Autonomous Decision Making

✅ Local LLM Support (Ollama)

✅ LangGraph Workflow

---

# 🧠 Architecture

```text
                    USER
                      │
                      ▼
        Conversation Manager
                      │
                      ▼
          Memory Retrieval
                      │
                      ▼
           Context Builder
                      │
                      ▼
         Goal Understanding
                      │
                      ▼
                Planner
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
    Tool Executor          Memory Manager
          │                       │
          ▼                       ▼
 Progress Evaluator         Save Memory
          │                       │
          ▼                       ▼
      Reflection               END
          │
          ▼
       Planner
```

---

# 🚀 Agent Workflow

```text
Goal

↓

Planning

↓

Tool Selection

↓

Execution

↓

Progress Evaluation

↓

Reflection

↓

Goal Complete?

YES → Save Memory → END

NO → Replan
```

---

# 🗂 Project Structure

```bash
LuxionV2/

├── graph.py
├── nodes.py
├── state.py
├── main.py
│
├── tools/
│   ├── registry.py
│   ├── web_tools.py
│   ├── file_tools.py
│   ├── response_tools.py
│   └── code_tools.py
│
├── memory/
│
├── prompts/
│
└── README.md
```

---

# ⚡ Core Components

| Component | Description |
|-----------|-------------|
| Conversation Manager | Handles user conversation |
| Memory Retrieval | Retrieves relevant memories |
| Context Builder | Builds contextual information |
| Goal Understanding | Understands user intent |
| Planner | Chooses next action |
| Tool Executor | Executes tools |
| Progress Evaluator | Tracks completed work |
| Reflection | Reviews execution quality |
| Memory Manager | Decides memory storage |
| Save Memory | Stores long-term memories |

---

# 🛠 Available Tools

- 🌐 Web Search
- 📄 File Writer
- 📂 File Reader
- 🧮 Python Execution
- 💬 LLM Response
- 📁 Memory Storage

---

# 💻 Installation

```bash
git clone https://github.com/articblue3452/LuxionV2.git

cd LuxionV2

pip install -r requirements.txt
```

---

# ▶ Run

```bash
python main.py
```

---

# Example

```text
You:

Find today's AI news
Summarize it
Save it into a file

━━━━━━━━━━━━━━━━━━━━━━

Goal Understanding ✓

Planning ✓

Web Search ✓

Summary ✓

File Created ✓

Memory Saved ✓

Done.
```

---

# 🧩 Technologies

- Python
- LangGraph
- Ollama
- Local LLMs
- JSON
- Tool Calling
- Agent Architecture

---

# 📈 Roadmap

## ✅ V1

- Basic Planner
- Tool Execution

---

## ✅ V2

- Memory
- Reflection
- Progress Tracking
- Goal Driven Planning

---

## 🚧 V3

- Multi-Agent Collaboration

- Tool Learning

- Self Improvement

- Better Planning

- Parallel Tool Execution

- Dynamic Skill Creation

---

## 🌌 Future Vision

- Recursive Self Improvement (RSI)

- Autonomous Software Engineer

- Robotics Integration

- Voice Interface

- Computer Control

- Vision Models

---

# 🎬 Demo

Replace this with a terminal GIF later.

```text
python main.py

Goal:
Find today's AI news

Planning...

Searching...

Summarizing...

Saving...

Done.
```

---

# 📊 Project Status

```
███████████████████░░░░░░░░ 70%

Architecture      ████████████

Planning          ███████████

Reflection        ███████████

Memory            ██████████

Tools             ██████████

Self Learning     ███
```


---

# 🌟 Support

If you like this project,

⭐ Star the repository.

🍴 Fork it.

💡 Contribute.

---

<p align="center">

### "The goal isn't to build another chatbot.

### The goal is to build an intelligent autonomous system."

</p>

---

<p align="center">

Made with ❤️ by **ArticBlue**

</p>