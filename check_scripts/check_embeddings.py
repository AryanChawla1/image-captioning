import pandas as pd
import numpy as np

FILEPATH = "embedded_data/flickr8k_train_embedded.parquet"

df = pd.read_parquet(FILEPATH)

def slice_embeds(embed_array):
    if isinstance(embed_array, np.ndarray):
        return embed_array[:5]
    return np.array([])

display_df = pd.DataFrame({
    'Caption': df['caption'],
    'Image_Embeds (First 5)': df['image_embeds'].apply(slice_embeds),
    'Text_Embeds (First 5)': df['text_embeds'].apply(slice_embeds)
})

print(f"Verification Sample (First 5 Rows) from: {FILEPATH}")
print(display_df.head(5))
print("\n(Note: The full embeddings contain 512 values each.)")