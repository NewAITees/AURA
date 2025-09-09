# AURA - Autonomous Unified Research Assistant

Project AURA is a modular, multi-agent AI coding framework that leverages local LLM infrastructure for secure, autonomous software development. The system employs a microservices architecture with containerized execution environments, enabling specialized AI agents to collaborate on complex coding tasks while maintaining security and observability.

## Features

- **Multi-Agent Architecture**: Specialized agents for code generation, review, testing, and integration
- **Local LLM Integration**: Uses Ollama for secure, local AI processing
- **Secure Execution**: Docker + Firejail sandboxing for safe code execution
- **Advanced TUI**: Rich terminal interface with real-time monitoring
- **Workflow Orchestration**: LangGraph-based agent coordination
- **High Performance**: Redis caching and SQLite persistence

## Architecture

```
TUI Interface ──→ Orchestration Manager ──→ Agent Scheduler
     ↓                    ↓                      ↓
Monitoring System ←── State Manager ←──── Specialized Agents
     ↓                    ↓                      ↓
System Health        Redis/SQLite       Execution Environment
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Ollama**
   ```bash
   ollama pull llama3.1:8b
   ollama pull mixtral:8x7b
   ```

3. **Start Redis**
   ```bash
   redis-server
   ```

4. **Run AURA**
   ```bash
   python -m aura.main
   ```

## Project Structure

```
aura/
├── agents/          # Specialized AI agents
├── core/            # Core orchestration and management
├── execution/       # Secure code execution environment
├── interface/       # TUI and user interface
├── models/          # LLM management and optimization
├── state/           # State management and persistence
├── security/        # Security profiles and monitoring
├── tests/           # Test suite
└── config/          # Configuration files
```

## Requirements

- Python 3.11+
- Docker
- Redis
- Ollama
- 16GB+ RAM (for larger models)
- NVIDIA GPU (recommended)

## Documentation

- [Design Document](.kiro/specs/autonomous-ai-coding-framework/design.md)
- [Requirements](.kiro/specs/autonomous-ai-coding-framework/requirements.md)
- [Tasks](.kiro/specs/autonomous-ai-coding-framework/tasks.md)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Security

AURA uses multiple security layers:
- Docker containerization
- Firejail sandboxing
- Network isolation
- Resource limits
- Security monitoring

Report security issues to the maintainers privately.