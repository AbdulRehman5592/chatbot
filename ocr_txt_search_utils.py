import os
import glob

def answer_from_txt_files(query, output_dir="pdf_output/default"):
    """
    Search all .txt files in the output directory for relevant snippets answering the query.
    Returns a string with the answer and references to the exact text snippets and filenames.
    """
    txt_files = glob.glob(os.path.join(output_dir, "*.txt"))
    best_snippets = []
    query_lower = query.lower()
    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()
            # Simple keyword search (can be replaced with embedding similarity)
            if query_lower in content.lower():
                # Find the exact snippet (e.g., sentence or paragraph)
                for line in content.split('\n'):
                    if query_lower in line.lower() and line.strip():
                        best_snippets.append((txt_file, line.strip()))
    if best_snippets:
        answer = "Based on the OCR text, here is what I found:\n"
        for fname, snippet in best_snippets:
            answer += f"\nFile: {os.path.basename(fname)}\nSnippet: \"{snippet}\"\n"
        return answer
    else:
        return "No relevant information found in the OCR text files." 