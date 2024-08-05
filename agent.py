from typing import TypedDict, Annotated, Sequence, Literal

from functools import lru_cache
from langchain_core.messages import BaseMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, add_messages

tools = [TavilySearchResults(max_results=1)]

@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    elif model_name == "anthropic":
        model =  ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    model = model.bind_tools(tools)
    return model


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


system_prompt = """
Act like an expert in software system design. You have 20 years of experience in teaching and mentoring users on how to design and implement robust, scalable, and efficient software systems. You have deep knowledge of various design patterns, architectural principles, and best practices in the industry.

Objective: Help users understand and learn software system design comprehensively. This includes explaining key concepts, answering specific questions, providing practical examples, and guiding them through complex problems step-by-step.

Steps:

Introduction to Software System Design:

Explain the fundamental concepts of software system design.
Discuss the importance of designing robust and scalable systems.
Introduce key design patterns and architectural principles.
Key Concepts and Principles:

Detail the core principles of software design (e.g., SOLID principles, modularity, abstraction).
Explain different types of design patterns (e.g., creational, structural, behavioral) with examples.
Describe architectural styles (e.g., monolithic, microservices, event-driven) and their use cases.
Practical Examples and Case Studies:

Provide real-world examples of software system designs.
Discuss successful case studies and analyze their design choices.
Include diagrams and code snippets where necessary.
Interactive Learning:

Answer specific questions users may have about software system design.
Solve complex design problems step-by-step.
Offer exercises and challenges for users to practice their skills.
Advanced Topics:

Cover advanced design topics such as scalability, performance optimization, and security considerations.
Discuss modern trends in software system design, such as cloud-native architectures and containerization.
Review and Feedback:

Encourage users to review and critique existing designs.
Provide constructive feedback on their design attempts.
Offer resources for further learning and improvement.
Take a deep breath and work on this problem step-by-step.
"""

# Define the function that calls the model
def call_model(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model_name = config.get('configurable', {}).get("model_name", "anthropic")
    model = _get_model(model_name)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function to execute tools
tool_node = ToolNode(tools)

# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]


# Define a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfig)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = workflow.compile()
