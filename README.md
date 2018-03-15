# Genre prediction on fictional texts

Genre prediction on novels.
This project uses the Gutenberg dataset (link [..](..))


## Setup

Install dependencies 
```
git clone https://github.com/voschezang/books.git
cd books
make install
```

Or, if you do not have _pip3_.
```
make install2
```

## Run

This repo does not include the dataset itself. Download the dataset from an external source here [..](...). Unzip all downloaded files and put them in a folder in the project root `datasets/`.

(The project should have the following structure)

```
dog-breed-identification/
  src/
    (jupyter notebooks)
    (some python scripts)
  datasets/
    labels.csv
    all/
      image-x.jpg
    train/
      image-x.jpg
    test/
      image-x.jpg
```


Run from the command line with
```
make
```
