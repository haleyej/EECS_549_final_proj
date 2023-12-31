from doc_augmentor import DocumentAugmentor
import json 
import jsonlines
import os
from tqdm import tqdm


def read_dataset(path: str, title_key:str, body_text_key: str = None) -> dict[int, str]:
    '''
    run augmentation on the entire corpus
    '''
    docid_to_text = {}
    with open(path) as f:
        doc = f.readline()
        while doc:
            doc = json.loads(doc)
            docid = doc['docid']
            post = doc[title_key]
            if body_text_key != None:
                post_body = doc[body_text_key]
                if post_body != '' or post_body != None:
                    post = post + ' ' + post_body
            docid_to_text[docid] = post
            doc = f.readline()

    return docid_to_text

def augment_dataset(docs: dict[int, str], stopwords: str = 'english') -> dict[int, str]:
    docids = list(docs.keys())
    augmentor = DocumentAugmentor(docs, stopwords)

    augmented_posts = augmentor.augment_docs(docids)

    return augmented_posts

def create_augmented_dataset(augmented_posts: dict[int, str], dataset_path: str, save_path: str) -> None:
    comments_data = read_dataset(dataset_path, title_key = 'text')
    docids = list(augmented_posts.keys())

    docid_to_aug_full_text = []
    for docid in tqdm(docids):
        augmentated_post = augmented_posts[docid]
        comments = comments_data[docid]

        docid_to_aug_full_text.append({"docid": docid, 
                                       "post": augmentated_post, 
                                       "comments": comments, 
                                       "full_text": augmentated_post + " " + comments})

    with jsonlines.open(save_path, 'w') as f: 
        f.write_all(docid_to_aug_full_text)


def main():
    os.chdir("code")
    path = "../data/ask_reddit_posts_v2.jsonl"
    docid_to_post = read_dataset(path, title_key = 'post', body_text_key = 'post_body') 

    augmented_dataset = augment_dataset(docid_to_post)
    create_augmented_dataset(augmented_dataset, '../data/ask_reddit_posts_v2.jsonl', '../data/ask_reddit_posts_augmented.jsonl')


if __name__ == '__main__':
    main()