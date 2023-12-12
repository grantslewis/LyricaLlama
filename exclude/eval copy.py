import json
import tkinter as tk
from tkinter import simpledialog

def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def save_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def display_results(window, data):
    for key, value in data.items():
        results = value.get('results', [])
        if len(results) >= 6:
            displayed_results = results[:6]

            # Create a frame for each result
            for i, result in enumerate(displayed_results):
                txt = result['responses']
                
                frame = tk.Frame(window, borderwidth=1, relief="solid")
                frame.grid(row=1, column=i, sticky="nsew")
                label = tk.Label(frame, text=f"Result {i} (Rank 0-5)", bg="lightgrey")
                label.pack(fill="x")

                text = '\n'.join(txt.split('\n')[:15])  # Display first 15 lines
                text_widget = tk.Text(frame, wrap="none", height=15, width=40)
                text_widget.insert("1.0", text)
                text_widget.configure(state="disabled")
                text_widget.pack(side="left", fill="both", expand=True)

            # Ranking input
            ranking_str = simpledialog.askstring("Input", "Enter rankings (comma-separated):", parent=window)
            rankings = [int(x.strip()) for x in ranking_str.split(',')]
            data[key]['rankings'] = rankings
            break  # Only display one key-value pair at a time

def main():
    #filename = input("Enter the filename of the JSON file: ")
    filename = "./combined_results.json"
    data = load_json(filename)

    window = tk.Tk()
    window.title("Results Ranking")
    display_results(window, data)

    window.mainloop()

    new_filename = filename.rsplit('.', 1)[0] + '_v2.json'
    save_json(data, new_filename)
    print(f"Updated data saved to {new_filename}")

if __name__ == "__main__":
    main()
