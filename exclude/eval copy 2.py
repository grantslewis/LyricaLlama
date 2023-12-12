import json
import tkinter as tk
from tkinter import ttk
import random

def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def save_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def display_results(window, data):
    for key, value in data.items():
        results = value.get('results', [])
        artist = value.get('artist', 'Unknown Artist')
        song = value.get('song', 'Unknown Song')
        genres = value.get('genres', [])

        if len(results) >= 6:
            #displayed_results = random.sample(results[:6], 6)
            ordered_res = results[:6]
            order = random.sample([i for i in range(len(ordered_res))], len(ordered_res))
            ordered_res = [ordered_res[i] for i in order]
            
            # Create frames for each result
            for i, result in enumerate(ordered_res):
                txt = result['responses']
                frame = tk.Frame(window, borderwidth=1, relief="solid")
                frame.grid(row=1, column=i, sticky="nsew")

                # Display artist, song, and genres
                info_text = f"Artist: {artist}\nSong: {song}\nGenres: {', '.join(genres) if genres else 'None'}"
                info_label = tk.Label(frame, text=info_text, bg="lightgrey")
                info_label.pack(fill="x")

                text = '\n'.join(txt.split('\n')[:15])  # Display first 15 lines
                text_widget = tk.Text(frame, wrap="none", height=15, width=40)
                text_widget.insert("1.0", text)
                text_widget.configure(state="disabled")
                text_widget.pack(side="left", fill="both", expand=True)

            # Radio buttons for ranking
            #rankings = [tk.IntVar() for _ in range(6)]
            
            
            
            #for i in range(6):
            #    frame = tk.Frame(window)
            #    frame.grid(row=2, column=i, sticky="nsew")
            #    for j in range(6):
            #        radio_button = ttk.Radiobutton(frame, text=str(j), variable=rankings[i], value=j)
            #        radio_button.pack(anchor='w')

            # Radio buttons for ranking
            rankings = [tk.IntVar() for _ in range(6)]
            for i in range(6):
                frame = tk.Frame(window)
                frame.grid(row=2, column=i, sticky="nsew")

                # Add labels for ordering
                #if i == 0:
                #    tk.Label(frame, text="Best to Worst", bg="lightgreen").pack(anchor='w')
                #elif i == 5:
                #    tk.Label(frame, text="Worst", bg="lightcoral").pack(anchor='w')

                for j in range(6):
                    print(f"i: {i}, j: {j}")
                    radio_button = ttk.Radiobutton(frame, text=f'#{1+j}', variable=rankings[j], value=i)
                    radio_button.pack(anchor='w')
                    # Set default value to j
                    if i == j:
                        radio_button.invoke()


            def submit():
                ranking_res = [var.get() for var in rankings]
                reordered = [ranking_res[i] for i in order]
                data[key]['rankings'] = reordered #[var.get() for var in rankings]
                window.quit()

            submit_button = tk.Button(window, text="Submit", command=submit)
            #submit_button.grid(row=3, column=0, columnspan=6)
            submit_button.grid(row=9, column=0, columnspan=6)
            window.mainloop()
            return data

def main():
    #filename = input("Enter the filename of the JSON file: ")
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