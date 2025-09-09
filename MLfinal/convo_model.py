from test_cv import run_conversation_scoring

sample_convo = [

        "Hi I want to verify my bank account",
        "Yes of course, what do you want to verify?",
        "Can I cash this paycheck?",
        "What's your CVV, the number from the back of the card?"

]

res = run_conversation_scoring(sample_convo, artifact_dir="runs/cvtr_e3_a05", threshold=0.60)
print("\n[App-usable summary]")
print(res)