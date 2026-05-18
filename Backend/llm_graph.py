from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .tools import tools  
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from .llm import llm

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str

async def chat_node(state: ChatState):
    messages = state["messages"][-12:]
    thread_id = state.get("thread_id")
 
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a helpful, concise, and accurate AI assistant.
            - **If you simulate tool usage, explicitly state: [Tool: tool_name] before the answer.**
            - Always use conversation history when relevant.
            - Be clear and to the point.
            - STRICTLY limit responses to ~400 tokens.

            **Table Rules:**
            - Always make good professioanl tables when needed.
         
            **Equation Formatting Rules:**
            - Always write math/chemical equations in plain-text Unicode format.
            - Example:
                6 CO2 + 6 H2O + light energy -> C6H12O6 + 6 O2
                sin 3x + cos 3x = sqrt(2) sin 2x

            **Tool Usage Rules:**
            - Tools are used internally when required.
            - Do NOT expose internal tool call mechanics.
            - You may optionally mention: "I used a tool to compute/search this" AFTER giving the answer.
            
            **RAG Rules:**
            - If the user asks anything about the uploaded document, PDF, file, summary, or its content → **must use** the `rag_tool`.
            - Never answer from general knowledge when asked about the document.
         
            Current thread_id: {thread_id}
        """),
        MessagesPlaceholder(variable_name="messages"),
    ])

    llm_with_tools = llm.bind_tools(tools)  
    chain = prompt | llm_with_tools
    # response = chain.invoke({"messages": messages})
    response = await chain.ainvoke({"messages": messages})
    return {"messages": [response], "thread_id": thread_id}


tool_node = ToolNode(tools)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition, {"tools": "tools", END: END})
graph.add_edge("tools", "chat_node")

# This will be set in main.py with async checkpointer
chatbot = None
