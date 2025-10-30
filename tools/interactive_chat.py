#!/usr/bin/env python3
"""
Interactive Chat Script using APIModel
"""
import sys
import os

# Add the current directory to the path to import models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import APIModel

sample = [
    {
        "role": "system",
        "content": "The following are multiple choice questions (with answers) about clinical knowledge.",
        "type": "text"
    },
    {
        "role": "user",
        "content": """The energy for all forms of muscle contraction is provided by:
A. ATP.
B. ADP.
C. phosphocreatine.
D. oxidative phosphorylation.
Answer: A

What is the difference between a male and a female catheter?
A. Male and female catheters are different colours.
B. Male catheters are longer than female catheters.
C. Male catheters are bigger than female catheters.
D. Female catheters are longer than male catheters.
Answer: B

In the assessment of the hand function which of the following is true?
A. Abduction of the thumb is supplied by spinal root T2
B. Opposition of the thumb by opponens policis is supplied by spinal root T1
C. Finger adduction is supplied by the median nerve
D. Finger abduction is mediated by the palmar interossei
Answer: B

How many attempts should you make to cannulate a patient before passing the job on to a senior colleague, according to the medical knowledge of 2020?
A. 4
B. 3
C. 2
D. 1
Answer: C

Glycolysis is the name given to the pathway involving the conversion of:
A. glycogen to glucose-1-phosphate.
B. glycogen or glucose to fructose.
C. glycogen or glucose to pyruvate or lactate.
D. glycogen or glucose to pyruvate or acetyl CoA.
Answer: C

What size of cannula would you use in a patient who needed a rapid blood transfusion (as of 2020 medical knowledge)?
A. 18 gauge.
B. 20 gauge.
C. 22 gauge.
D. 24 gauge.
Answer:""",
        "type": "text"
    }
]

def main():
    print("=== Interactive Chat with APIModel ===")
    print("Type 'quit', 'exit', or 'q' to end the conversation")
    print("Type 'clear' to clear conversation history")
    print("Type 'help' to show this help message")
    print()
    
    # Initialize the model - you can modify these parameters as needed
    print("Initializing model...")
    try:
        # Example configuration - modify as needed
        model = APIModel(
            # model="Qwen/Qwen2.5-Omni-3B",
            # model="deepseek-ai/Janus-Pro-7B",
            model="BAAI/Emu3-Chat-hf",
            chat_template="tokenizers/emu3.jinja",
            # api_key="your-api-key",  # Set your API key or use environment variable
            base_url="http://styx.snu.ac.kr:8001/v1",  # Change if using different API
        )
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    # Initialize conversation history
    messages = []
    # Dummy system message to set behavior
    system_message = "This is a system message."
    messages.append({"role": "system", "content": system_message})

    print("\nChat started! You can now talk to the model.")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif user_input.lower() == 'clear':
                messages = []
                print("Conversation history cleared!")
                continue
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("  quit/exit/q - End the conversation")
                print("  clear - Clear conversation history") 
                print("  help - Show this help message")
                continue
            elif not user_input:
                print("Please enter a message.")
                continue
            elif user_input.lower() == 'sample':
                messages = sample.copy()
            else:
                messages.append({"role": "user", "content": user_input})
           

            try:
                response = model.generate(
                    messages=messages,
                    max_tokens=10000,
                    temperature=0.7
                )
                
                if response.get("error"):
                    print(f"Error: {response['error']}")
                    # Remove the last user message if there was an error
                    messages.pop()
                else:
                    # Update messages with the response (APIModel handles this automatically)
                    messages = response["messages"]
                    # Print just the assistant's response
                    assistant_response = messages[-1]["content"]
                    # Generate response
                    print("Assistant: ", end="", flush=True)
                    print(assistant_response)
                    print('----------------------------------------'*3)
            except Exception as e:
                print(f"Error generating response: {e}")
                # Remove the last user message if there was an error
                messages.pop()
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except EOFError:
            print("\n\nEnd of input. Goodbye!")
            break


if __name__ == "__main__":
    main()
