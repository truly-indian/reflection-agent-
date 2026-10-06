from dotenv import load_dotenv
from typing import List, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from chains import generation_chain, reflection_chain

REFLECT = "reflect"
GENERATE = "generate"

load_dotenv()

def generation_node(state: Sequence[BaseMessage]):
    return generation_chain.invoke({"messages": state})

def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    result = reflection_chain.invoke({"messages": messages})
    return [HumanMessage(content=result.content)]

builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: List[BaseMessage]):
    print("\nTweet:", state[-1].content)
    answer = input("\nDo you like this tweet? (yes/no): ").strip().lower()
    if answer == "yes":
        return END
    elif answer == "no":
        return REFLECT
    else: 
        print("Please enter 'yes' or 'no'.")
        return should_continue(state) 

builder.add_conditional_edges(GENERATE, should_continue, {END: END, REFLECT: REFLECT})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
#print(graph.get_graph().draw_mermaid())

def main():
    inputs = HumanMessage(content="""Hi, Please make my below tweet better: 
                            Today the wether is suny. 
                          """)
    response = graph.invoke(inputs)
    
    print("\nFinal tweet:")
    print(response[-1].content)
    
    
    
if __name__ == "__main__":
    main()
