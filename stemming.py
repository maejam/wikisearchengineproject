from nltk.stem import SnowballStemmer


suffixes = ["s", "ing", "ed", "ial"]

def naive_stemmer(token: str):
    for suffix in suffixes:
        if token.endswith(suffix):
            newtok = token[:-len(suffix)]
            break
        else:
            newtok = token
    return newtok

stemmer = SnowballStemmer("english")

if __name__ == "__main__":
    l=["bacterial", "walked", "bacterias", "walking"]
    print("Naive\tSnowball")
    print("---------------")
    for word in l:
        print(naive_stemmer(word), "\t", stemmer.stem(word))
