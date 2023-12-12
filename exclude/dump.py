
# model_name = "grantsl/LyricaLlama"
models_to_test = ["meta-llama/Llama-2-7b-hf", "grantsl/LyricaLlama"]

CURRENT_IND = 1

RESP_COUNT = 2 #3
MAX_LEN = 500

# for model_name in models_to_test:
model_name = models_to_test[CURRENT_IND]


model_ver = os.path.basename(model_name)
print(model_ver)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)



test_set = [{'is_new':False, 'song':'All I Want For Christmas', 'artist':'Mariah Carey', 'genres':['Christmas']},
            {'is_new':False, 'song':'Blinding Lights', 'artist':'The Weeknd', 'genres':['Pop', 'Synth-Pop']},
            {'is_new':False, 'song':'Lover', 'artist':'Taylor Swift', 'genres':['Pop']},
            {'is_new':True, 'song':'Electric Touch', 'artist':'Taylor Swift and Fall Out Boy', 'genres':['Pop', 'Pop/Rock']},
            {'is_new':True, 'song':'Next Thing You Know', 'artist':'Jordan Davis', 'genres':['Country']},
            {'is_new':True, 'song':'Penthouse', 'artist':'Kelsea Ballerini', 'genres':['Pop', 'Country/Pop']},
           ]



results = dict()

for song_info in tqdm(test_set):
    res_song = song_info.copy()
    # print(song_info.keys())
    song = song_info['song']
    artist = song_info['artist']
    genres = song_info['genres']

    inputs = generate_input(song, artist, genres)
    inputs = generate_text(inputs)

    inputs_len = len(inputs)

    responses = []
    for i in range(RESP_COUNT):
        response = generate_response(inputs, tokenizer, model, min(MAX_LEN + inputs_len, 4096))

        response_cleaned = response[inputs_len:]
        responses.append(response_cleaned)

    res_song['responses'] = responses

    results[song] = res_song

with open(f'./results_{model_ver}.json', 'w') as f:
    json.dump(results, f, indent=4)
    
del model

gc.collect()
torch.cuda.empty_cache() 