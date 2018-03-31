# Genre prediction

Genre prediction on novels. This project uses the Gutenberg dataset.

This repository does not include the dataset itself. Download the dataset [here](https://drive.google.com/file/d/1J2gBeq8sOdePKzx8_VZ46PRm7-TcMh9B/view?usp=sharing).

## Run

Install dependencies 
```
git clone https://github.com/voschezang/books.git
cd books
make deps
```

Run the predictor. `_book_` should be a _.txt_ file. E.g. _`book=datasets/test/12.txt`_.

```
make predict book=~mybook.txt
```



---

If you do not have _pip3_.
```
make deps2
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
