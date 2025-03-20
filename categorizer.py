import ollama
import os

model= "llama3.2"

input_file = "Ollama\Data\grocery_list.txt"
output_file = "Ollama/Data/grocery_list_categorized.txt"

if not os.path.exists(input_file):
    print(f"Input file {input_file} not found")
    exit(1)

with open(input_file, 'r') as f:
    items = f.read().strip()

prompt= f""" You are an assistant that categorizes and sorts grocery items. Here is a list of grocery items:
{items}

please: 
1. Categorize the items into appropriate categories such as fruits, vegetables, dairy, etc.
2. Sort the items alphabetically within each category.
3. Present the categorized list in a clear and organized manner.
4. translate the items into arabic.
"""

try:
    response= ollama.generate(model=model, prompt=prompt)
    generated_text= response.get("response", "")
    print("*** Generated categorized list ***")
    print(generated_text)
    with open(output_file, 'w') as f:
        f.write(generated_text.strip())
    print(f"Generated categorized list saved to {output_file}")
except Exception as e:
    print(f"Error: {str(e)}")
