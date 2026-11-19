import json
import sys

from app import rag

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or input("Question: ")
    print(json.dumps(rag.answer_question(q), indent=2))
