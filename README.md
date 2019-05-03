[Try out the Google Colab notebook here!](https://colab.research.google.com/drive/10WS1AgyL8DOUw14TFzkvZjn6aBKd5KYN)

# Fair-Fashion-MNIST

Fair-Fashion-MNIST proposes a modified [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset with very similar images removed. 

Download the dataset here:
- [train-images-idx3-ubyte.gz](https://drive.google.com/file/d/1bc2Uzub_pF6Oj5cO656_YyqBLPpr305X/view?usp=sharing)
- [t10k-images-idx3-ubyte.gz](https://drive.google.com/file/d/1qLUOoUf-ysdVoaGKN8jElQlvDd5cn5lM/view?usp=sharing)
- [train-labels-idx1-ubyte.gz](https://drive.google.com/file/d/1GWynk44xSOK5KIVHOETN8ZJeYmXdP8YM/view?usp=sharing)
- [t10k-labels-idx1-ubyte.gz](https://drive.google.com/file/d/1y6I-qErtz4cgZR803BfqFlwK3mIwcE58/view?usp=sharing)


Full training results can be found [here](https://vigilant-hopper-6d70bf.netlify.com/)

### Replicating result
1. [Run train_sort_save.ipynb in Google Colab](https://colab.research.google.com/drive/10WS1AgyL8DOUw14TFzkvZjn6aBKd5KYN)
2. Move dup[0-9].pickle files saved in your Google Drive to /dups
3. Label images using label_dups.py
4. Remove the similar images with remove_similar.py
5. Lastly, run final_dataset.py to generate new idx-ubyte files
