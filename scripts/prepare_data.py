import struct
import os

# Sample basic phrases and prompts for our corpus
PHRASES = [
    "Hello, how are you today?",
    "I am a large language model.",
    "The weather is very nice.",
    "Can you help me with my homework?",
    "What is the capital of France?",
    "The capital of France is Paris.",
    "Tell me a story about a brave knight.",
    "Once upon a time, there was a knight named Sir Lancelot.",
    "He was known for his courage and honor.",
    "One day, he went on a quest to find a hidden treasure.",
    "The treasure was guarded by a fierce dragon.",
    "Sir Lancelot fought the dragon and won.",
    "He returned to the kingdom with the treasure.",
    "Everyone cheered for the brave knight.",
    "They lived happily ever after.",
    "The end.",
    "Goodbye!"
]

def prepare_data(output_txt="corpus.txt", output_bin="corpus.bin"):
    print(f"Creating {output_txt}...")
    with open(output_txt, "w") as f:
        for phrase in PHRASES:
            f.write(phrase + "\n")

    print(f"Tokenizing and creating {output_bin}...")
    # For now, we use a simple byte-level encoding for the binary file
    # as the Dataset class in C++ expects uint16_t (2 bytes) per token.
    # The C++ Tokenizer class can also be used to train and save a vocab.
    with open(output_txt, "r") as f:
        text = f.read()

    with open(output_bin, "wb") as f:
        for char in text:
            # Pack each byte as a uint16_t (little-endian)
            f.write(struct.pack("<H", ord(char)))

    print("Done!")

if __name__ == "__main__":
    prepare_data()
