import helper_funcs as hf

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
           'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 
           't', 'u', 'v', 'w', 'x', 'y', 'z']


batch_time = int(3e6) # Batches of 3 seconds of recording


for l in letters:
    hf.generate_datasets_csv(l, batch_time)
