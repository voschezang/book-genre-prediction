# Genre prediction

Genre prediction on novels. This project uses the Gutenberg dataset.

This repository does not include the dataset itself. If you want to train your own model, download the dataset [here](https://drive.google.com/file/d/1iO-skvTyxQ0YnVUNfoRC7nMLtXHDxNRl/view?usp=sharing).

The code has been tested on macOS High Sierra. 

## Run

Make sure _python3_ and _pip3_ are installed. All dependencies will be installed by pip.

```
git clone https://github.com/voschezang/books.git
cd books
make deps
```

Run the predictor. `_book_` should be a _.txt_ file. E.g. _`book=datasets/test/1118.txt`_.

```
make predict book=~mybook.txt
```



---

If you do not have _pip3_, you can try:
```
make deps2
python src/main.py mybook
```



---

(The project should have the following structure)

```
books/
  src/
    (jupyter notebooks)
    (some python scripts)
  datasets/
    models/
      (a pretrained model)
    labels.csv
    (other files)
```
