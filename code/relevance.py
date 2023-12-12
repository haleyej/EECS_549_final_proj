import csv 
import numpy as np 
from tqdm import tqdm
from pandas import DataFrame
import pandas as pd
from ranker import Ranker, BM25
import pickle
import os

def map_score(search_result_relevances: list[int], cut_off=10) -> float:
    precision = []
    cut_off = min(len(search_result_relevances), cut_off)
    for i in range(len(search_result_relevances[:cut_off])):
        part = search_result_relevances[:(i + 1)]
        if part[i] == 0:
            precision.append(0)
            continue
        num_pos = part.count(1.0)
        precision.append(num_pos / (i + 1))

    if np.sum(precision) == 0:
        return 0
    map = np.sum(precision) / cut_off
    return map

def ndcg_score(search_result_relevances: list[float], 
               ideal_relevance_score_ordering: list[float], cut_off=10):
    actual_rating = []
    ideal_rating = []
    cut_off = min(len(ideal_relevance_score_ordering), len(search_result_relevances), cut_off)
    for i in range(cut_off):
        rating = search_result_relevances[i]
        ideal_val = ideal_relevance_score_ordering[i]
        if i == 0:
            actual_rating.append(rating)
            ideal_rating.append(ideal_val)
        elif i > 0:
            rating = rating / np.log2(i + 1)
            ideal_val = ideal_val / np.log2(i + 1)
            actual_rating.append(rating)
            ideal_rating.append(ideal_val)

    if np.sum(ideal_rating) > 0:
        return np.sum(actual_rating) / np.sum(ideal_rating)
    else:
        return 0


def map_queries_to_judgements(relevance_data_filename:str) -> dict[str, list[tuple[int]]]:
    with open(relevance_data_filename) as f:
            reader = csv.reader(f)
            queries_to_judgements = {}
            header = next(reader)
            query_idx = header.index('query')
            rel_idx = header.index('rel')
            docid_idx= header.index('docid')
            for line in tqdm(reader): 
                query = line[query_idx]
                rel = int(line[rel_idx])
                docid = int(line[docid_idx])
                queries_to_judgements[query] = queries_to_judgements.get(query, []) + [(docid, rel)]
    return queries_to_judgements


def run_relevance_tests(queries_to_judgements: dict, outfile:str, ranker, cutoff:int = 10) -> dict[str, float]:
    maps = []
    ncdgs = []
    for query, relevance_ratings in list(queries_to_judgements.items()):
        search_results = ranker.query(query)
        map_relevance_scores = []
        ndcg_relevance_scores = []
        # add zero for ones 
        if len(search_results) == 0:
            maps.append(0)
            ncdgs.append(0)
            continue
        for result in search_results[:cutoff]:
            if len(result) == 0:
                map_relevance_scores.append(0)
                ndcg_relevance_scores.append(0)
                continue
            result_docid = result[0]
            found = False
            for tup in relevance_ratings:
                docid, rel  = tup
                if int(docid) == int(result_docid):
                    found = True
                    if int(rel) >= 4:
                        map_relevance_scores.append(1)
                        ndcg_relevance_scores.append(1)
                        break
                    else:
                        map_relevance_scores.append(0)
                        ndcg_relevance_scores.append(0)
                        break
            if not found:
                map_relevance_scores.append(0)
                ndcg_relevance_scores.append(0)

        map = map_score(map_relevance_scores, cutoff)
        ncdg = ndcg_score(map_relevance_scores, sorted(ndcg_relevance_scores, reverse = True), cutoff)

        maps.append(map)
        ncdgs.append(ncdg)

        if outfile != '':
            with open(outfile, "a") as e:
                writer = csv.writer(e, delimiter=",")
                r1 = [query, 'map', map]
                r2 = [query, 'ndcg', ncdg]
                writer.writerow(r1)
                writer.writerow(r2)
    return {'map': np.mean(maps), 'ncdg': np.mean(ncdgs)}

