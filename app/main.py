from rag_setup import ask, run, memory

print(ask("How far in advance must I request annual leave longer than five working days?"))
print(ask("And how do I submit it?"))

# 1. Single questions with known answers
run([
    "How long do I have to return a product?",                     # 30 calendar days
    "How long does a card refund usually take?",                             # 3-7 business days
    "What is the price of the Nordview NQ-55U7 TV?",                         # €649.00
    "Within how many days must expense claims be submitted, and in which system?",  # 30 calendar days, SpendHub
    "How quickly must a manager respond to a remote work request?",          # 15 calendar days
])

# 2. Follow-ups: these test the memory
memory.clear()
run([
    "How long does a card refund usually take?",
    "And what about bank transfers?",              # 3-5 business days after bank details validated
    "Which of the two is faster?",                 # answered from history, bank transfer or card
])

memory.clear()
run([
    "What equipment does IronStore provide for remote work?",
    "Who approves a remote work request?",
    "what did i ask?",
])

memory.clear()
run([
    "How long do I have to return a product?",
    "Does that apply to personalised goods too?",     # "that" = the 30-day return window
])
