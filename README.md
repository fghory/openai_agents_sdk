# Unleashing AI Agents: Exploring OpenAI Agent SDK

- This repository contains a collection of examples demonstrating the capabilities of the OpenAI Agents SDK, showcasing various implementations of AI agents, voice interactions, and specialized assistants. 
- Notably, examples given in OpenAI Agents SDK repository has been customized.
- Gemini models have been mostly used for generous free tier availability though some feature implementation strictly requirs OPenAI models. AgentOps has been used for tracing and Tavily Seach for web search as it also offers free tier.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- UV package manager
- OpenAI API key
- Gemini API key
- Tavily API key
- AgentOps API Key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd openai_agents_sdk
```

2. Install dependencies using UV:
```bash
uv init --package openai-agents-sdk
uv add openai-agents[voice]  # For voice-enabled examples
```

3. Set up your environment variables:
```bash
# Create a .env file
cat <<EOF > .env
OPENAI_API_KEY=your-openai-key-here
GEMINI_API_KEY=your-gemini-key-here
TAVILY_API_KEY=your-tavily-key-here
AGENTOPS_API_KEY=your-agentops-key-here
EOF
```

## 📁 Project Structure

The repository contains several example implementations:

### Core Examples
- `first_agent/` - Basic agent implementation
- `first_handoff/` - Example of agent handoff functionality
- `chain_hello/` - Simple Hello Program 

### Specialized Assistants
- `research_assistant/` - AI-powered research assistant
- `financial_research_assistant/` - Specialized financial research agent
- `travel_planner/` - Travel planning assistant
- `voice_agent/` - Voice-enabled agent implementation

### Learning Examples
- `exmaples_openai_agents/` - OpenAI Agents SDK repository examples
- `learning/` - Educational examples and tutorials

## 🎯 Key Features

### Agent Capabilities
- Built-in agent loop for tool calling and LLM interaction
- Python-first approach for agent orchestration
- Handoff functionality between agents
- Input validation and guardrails
- Function tools with automatic schema generation
- Built-in tracing for visualization and debugging


## 🛠 Usage Examples

### Basic Agent
```python
from agents import Agent, Runner

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant"
)

result = Runner.run_sync(agent, "Your query here")
print(result.final_output)
```


```

## 🔧 Development

### Code Style
- Follow PEP 8 guidelines for Python code
- Use type hints and docstrings
- Run linters before committing

```


## 📚 Documentation

For detailed documentation, visit:
- [OpenAI Agents SDK Documentation](https://openai.github.io/openai-agents-python/)
- [AgentOps OpenAI Agents Documentation](https://docs.agentops.ai/v1/integrations/agentssdk)



## ⚠️ Important Notes

- Always keep your API keys secure
- Monitor your API usage
- Follow OpenAI's usage guidelines
- Test thoroughly before deployment
