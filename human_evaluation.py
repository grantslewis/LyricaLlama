import json
import tkinter as tk
from tkinter import simpledialog
import random
import signal
import sys

# A program meant to allow a human to rank the results of the 2 models


def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def save_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def update_content(window, frame, key, value, data, order):
    results = value.get('results', [])
    artist = value.get('artist', 'Unknown Artist')
    song = value.get('song', 'Unknown Song')
    genres = value.get('genres', [])

    if len(results) >= 6:
        ordered_res = results[:6]
        #order = random.sample(range(len(ordered_res)), len(ordered_res))
        order = [i for i in range(len(ordered_res))]
        ordered_res = [ordered_res[i] for i in order]
        
        # Clear existing content
        for widget in frame.winfo_children():
            widget.destroy()

        # Create column headers
        for i in range(6):
            header = tk.Label(frame, text=f"#{i+1}", bg="lightgrey")
            header.grid(row=0, column=i, sticky="nsew")

        # Create frames for each result
        for i, result in enumerate(ordered_res):
            txt = result['responses']
            result_frame = tk.Frame(frame, borderwidth=1, relief="solid")
            result_frame.grid(row=1, column=i, sticky="nsew")

            # Display artist, song, and genres
            info_text = f"Artist: {artist}\nSong: {song}\nGenres: {', '.join(genres) if genres else 'None'}"
            info_label = tk.Label(result_frame, text=info_text, bg="lightgrey")
            info_label.pack(fill="x")

            text = '\n'.join(txt.split('\n')[:15])  # Display first 15 lines
            text_widget = tk.Text(result_frame, wrap="none", height=15, width=40)
            text_widget.insert("1.0", text)
            text_widget.configure(state="disabled")
            text_widget.pack(side="left", fill="both", expand=True)
        return order

def display_results(window, data):
    frame = tk.Frame(window)
    frame.pack(expand=True, fill="both")
    
    order = []
    next_pair = tk.BooleanVar(value=False)

    for key, value in data.items():
        order = update_content(window, frame, key, value, data, order)

        # Entry widget for rankings
        ranking_label = tk.Label(window, text="Enter your rankings (comma-separated):")
        ranking_label.pack()
        ranking_entry = tk.Entry(window)
        ranking_entry.pack()

        def submit():
            ranking_str = ranking_entry.get()
            rankings = [int(x.strip()) - 1 for x in ranking_str.split(',')]  # Adjust for zero-indexing
            reordered = [order[i] for i in rankings]
            data[key]['rankings_str'] = ranking_str
            data[key]['order'] = order
            data[key]['rankings'] = reordered
            next_pair.set(True)

        submit_button = tk.Button(window, text="Submit", command=submit)
        submit_button.pack()
        
        frame.wait_variable(next_pair)
        next_pair.set(False)
        submit_button.pack_forget()
        ranking_entry.pack_forget()
        ranking_label.pack_forget()

    window.quit()
    return data

def signal_handler(sig, frame):
    print("You pressed Ctrl+C! Exiting gracefully.")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    filename = "./combined_results.json"
    data = load_json(filename)

    window = tk.Tk()
    window.title("Results Ranking")
    updated_data = display_results(window, data)

    new_filename = filename.rsplit('.', 1)[0] + '_v2.json'
    save_json(updated_data, new_filename)
    print(f"Updated data saved to {new_filename}")

if __name__ == "__main__":
    main()
