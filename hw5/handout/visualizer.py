import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def draw_subgraph(data,index,letters):
    row_data = data.iloc[index]
    label = letters[row_data[0]]
    image = np.array(row_data[1:]).reshape(16, 8)
    return image,label
    

def draw(input_file,index1,index2,index3,save_path = None):
    
    # Load CSV file
    data_input = pd.read_csv(input_file, header=None)
    letters = {0: "a", 1: "e", 2: "g", 3: "i", 4: "l", 5: "n", 6: "o", 7: "r", 8: "t", 9: "u"}

    # Select specific row you want to look at in the file
    # Recall that the index should be the row number in the file minus 1

    image1,label1 = draw_subgraph(data_input,index1,letters)
    image2,label2 = draw_subgraph(data_input,index2,letters)
    image3,label3 = draw_subgraph(data_input,index3,letters)
    
    # Plot the image
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), gridspec_kw={'hspace': 0.1})  # hspace controls vertical gap

    # Display images with labels
    axes[0].imshow(image1, cmap='gray', interpolation='nearest')
    axes[0].set_title(f"Label: {label1}")
    axes[0].axis('off')
    
    axes[1].imshow(image2, cmap='gray', interpolation='nearest')
    axes[1].set_title(f"Label: {label2}")
    axes[1].axis('off')
    
    axes[2].imshow(image3, cmap='gray', interpolation='nearest')
    axes[2].set_title(f"Label: {label3}")
    axes[2].axis('off')
    
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

def main():
    input_file = sys.argv[1]
    index1 = int(sys.argv[2])
    index2 = int(sys.argv[3])
    index3 = int(sys.argv[4])
    save_path = sys.argv[5]
    draw(input_file,index1,index2,index3,save_path)


if __name__ == '__main__':
    main()

