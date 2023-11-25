from doc_augmentor import DocumentAugmentor
import json 


def read_dataset(path: str) -> dict[int, str]:
    '''
    run augmentation on the entire corpus
    '''
    docid_to_post = {}
    with open(path) as f:
        doc = f.readline()
        while doc:
            doc = json.loads(doc)
            docid = doc['docid']
            post = doc['post']
            docid_to_post[docid] = post
            doc = f.readline()

    return docid_to_post

def augment_dataset(docs: dict[int, str], stopwords: list[str], save_path: str) -> None:
    docids = list(docs.keys())
    augmentor = DocumentAugmentor(docs, stopwords)

    augmented_posts = augmentor.augment_docs(docids)

    with open(save_path, "w") as f: 
        for k, v in list(augmented_posts.items()):
            d = {'docid':k, 'text': v}
            json.dump(d, f)

def main():
    path = "../data/ask_reddit_posts_v2.jsonl"
    docid_to_post = read_dataset(path) 
    
    with open("../files/stopwords.txt") as f: 
        stopwords = f.readlines()

    augment_dataset(docid_to_post, stopwords, '..data/augmented_ask_reddit_post.jsonl')


if __name__ == '__main__':
    main()