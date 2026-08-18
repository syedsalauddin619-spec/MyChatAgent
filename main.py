#-------------------------------------------------------------------------------
# Name:        module5
# Purpose:
#
# Author:      User
#
# Created:     18-08-2026
# Copyright:   (c) User 2026
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import ollama

print("==============================")
print("      MY AI CHAT AGENT")
print("==============================")
print("Type 'exit' to stop.")

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "user", "content": user_input}
        ]
    )

    print("\nAI:", response["message"]["content"])