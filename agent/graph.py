from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, MessagesState, StateGraph
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from langserve import add_routes
from pydantic import BaseModel, Field
from typing import List, Union

# REPLACE WITH YOUR LLM URL API
url = "REPLACE_WITH_YOUR_LLM_API/v1"

llm = ChatOpenAI(
    base_url=url, # This is the URL of your local server
    api_key="token-abc123", # This is a random token. If you haven't set any token for your API, this will be ignored
    model="antoinekrajnc/customer-success-assistant",  # Your HuggingFace hosted model
    temperature=0.3,
    streaming=True # This is important to keep it to True if you want to implement streaming capabilites to your app
)

# This define LangGraph classic graph
workflow = StateGraph(state_schema=MessagesState)

# Define a function that will call the model
def call_model(state: MessagesState):

    # Refine formatted message construction to reduce unnecessary template markers
    formatted_messages = []
    
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            # Simplified format for Instruction and Input without redundant symbols
            instruction = f"Instruction:\n{message.content.strip()}"
            input_context = (
                f"Input:\n{state['user_context']}" if 'user_context' in state else "Input:\n"
            )
            formatted_messages.append(f"{instruction}\n{input_context}\nResponse:")
        elif isinstance(message, AIMessage):
            # Just append the assistant's response directly
            formatted_messages.append(message.content.strip())

    # Join all formatted messages for the final prompt content
    prompt_content = "\n".join(formatted_messages)
    
    # Use a SystemMessage, then add the full formatted content as the HumanMessage
    start_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a telecom assistant made to answer customer's requests"),
        HumanMessage(content=prompt_content)
    ])
    
    # Chain the system prompt with the LLM model
    model = start_prompt | llm | StrOutputParser()

    # Invoke the model with the properly formatted prompt content
    response = model.invoke({"messages": state["messages"]})
    return {"messages": response}

# This part defines the whole graph architecture
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# This is stored in RAM of the computer. 
# This is not ideal for production and large conversation history where you will need to define a PostgreSQL DB (more on the later on)
memory = MemorySaver()

# Store the whole graph 
graph = workflow.compile(checkpointer=memory)


# IF YOU WANTED TO BUILD A SYNCHRONOUS 
# YET SIMPLE API
 
# app = FastAPI(
#     title="Sample LangServe API",
#     version="0.1",
#     description="Simple FastAPI app that integrates LangServe"
# )

# class InputChat(BaseModel):
#     """Input for the chat endpoint."""

#     messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
#         ...,
#         description="The chat messages representing the current conversation.",
#     )

# add_routes(
#     app, # this is your FastAPI instance
#     graph.with_types(input_type=InputChat), # This is the Runnable chain that we defined above
#     path="/infer", # This is the endpoint 
#     enable_feedback_endpoint=True,
#     enable_public_trace_link_endpoint=True

# )

# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)