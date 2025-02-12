
"""Writes the error rates for the 39 datasets into a csv file.'''

This script reads in the submission file, which contains the top 100 
suggested IDs for each query image in each of the 39 photo-ID datasets, then
classifies each prediction as a false reject or false accept. The classification
depends on the strategy. After classifying the predictions, the script 
calculates the false accept and false reject rates for each dataset under each 
strategy and writes them to a csv file.
"""

from pathlib import Path

import pandas as pd
import numpy as np


def main():
    '''Write the error rates for the 39 datasets into a csv file.'''

    PRED_COUNT = np.insert(np.arange(5, 51, 5), 0, 1)
    P_MAX = 0.8
    P_MIN = 0.4

    # directories
    base_dir = Path.home()

    # read in the mapping file 
    map_path = f'{base_dir}/datasets/hw/raw_data/full_file_mapping.csv'
    mapping = read_mapping(map_path)

    # add in the submission results
    submission_path = 'input/submissions/rist-1000.csv'
    submission = pd.read_csv(submission_path)

    # merge the results and mapping
    results = merge_results(submission, mapping)

    # get rates
    rates = get_rates(results, PRED_COUNT)

    # get cost
    cost = get_cost(results, PRED_COUNT)

    # merge cost with rates
    rates = rates.merge(cost)

    # get metadata
    folder_specs = get_folder_specs(mapping, p_max=P_MAX, p_min=P_MIN)

    # add in ids for catalog
    id_mapping = mapping[['species', 'folder']].drop_duplicates().sort_values(['species', 'folder'])
    id_mapping.rename(columns={'folder':'catalog'}, inplace=True)
    id_mapping['catalog_id'] = get_catalog_ids(id_mapping.species)

    # add rates to metadata
    rates = id_mapping.merge(folder_specs).merge(rates)

    # export to csv 
    path = 'input/rates.csv'        
    rates.to_csv(path, index=False)

    return None

def read_mapping(map_path):
    """read in the mapping data and correct known errors."""

    # read in the full file mapping, correcting the known species errors
    mapping = pd.read_csv(map_path, dtype='string')
    mapping['species'] = (
        mapping['species'].replace(
            {"globis": "short_finned_pilot_whale",
             "pilot_whale": "short_finned_pilot_whale",
             "kiler_whale": "killer_whale",
             "pantropic_spotted_dolphin": "spotted_dolphin",
             "bottlenose_dolpin": "bottlenose_dolphin"}
        )
    )

    # correct the bottlenose/spinner error
    is_under = mapping['folder'] == 'Botlenose-Dolphin-Underwater'
    mapping.loc[is_under, 'species'] = 'spinner_dolphin'

    mapping = mapping.loc[mapping.folder != 'AET-Frasiers Dolphin']
    mapping = mapping.loc[mapping.folder != 'Bottlenose-Dolphin-NDD']

    return mapping

def merge_results(submission, mapping):
    '''Join the submission file to the mapping file.'''

    # rename and subset columns for easy merging
    submission.columns = ['image', 'predictions']
    results = mapping[['folder', 'img_name_new', 'new_individual_id']]
    results.columns = ['catalog', 'image', 'id']

    # merge
    results = results.merge(submission)

    return results

def get_rates(results, pred_count):
    """calculate error rates for each folder"""

    # results['new_individual_id'] = replace_bad_ids(results)

    # calculate error rate based on counts checked
    ctl = []
    for count in pred_count:
        
        # select labels top count predictions from submission
        preds = results.predictions.str.split(' ')
        preds = [p[:count] for p in preds]
        labs = results.id
    
        # classify each test image's predictions as TR, TA, FR, FA
        pred_class = [classify_prediction(p, l) for p, l in zip(preds, labs)]
    
        # name the column
        nm = f'pred_class_{count}'
        results[nm] = pred_class
    
        # calculate the rates of each error
        cross_tab = pd.crosstab(
            results['catalog'],
            results[nm],
            margins=False,
            normalize='index'
        )
    
        # for higher pred counts there are no false accepts, add column in
        if not 'FA' in cross_tab.columns:
            cross_tab['FA'] = 0.
    
        # subset the results
        df = cross_tab[['FR', 'FA']].reset_index()
        df['matches_checked'] = count
        
        # add to list to be concatenated
        ctl.append(df)
    
    # concatenate data frames
    rates = (
        pd.concat(ctl)[['catalog', 'matches_checked','FR','FA']]
          .sort_values(['catalog', 'matches_checked'])
          .reset_index(drop=True)
    )
    
    return rates 

def get_cost(results, pred_count):
    '''Calculate the cost of each action for each folder.'''

    # add in the label indices
    results['label_idx'] = get_label_index(results)

    ll = []
    # what's the number of ids that we'd have to evaluate for each action?
    for a in pred_count:

        # the cost of the first action is 0
        if a == 1:
            
            tmp_df = results.catalog.drop_duplicates().to_frame()
            tmp_df['cost'] = 0
            tmp_df['matches_checked'] = a
            
            ll.append(tmp_df)
            continue
        
        # how many ids would we evaluated?
        l = np.where(
            results.label_idx < a,
            results.label_idx + 1,
            a
        )

        # create a temporary dataframe to split/apply
        tmp_df = results.copy()
        tmp_df['cost'] = l
        tmp_df['matches_checked'] = a
        
        # the split/apply step
        tmp_df = tmp_df.groupby(['catalog', 'matches_checked'])['cost'].sum().reset_index()
        ll.append(tmp_df)

    # the combine step
    cost = pd.concat(ll)

    return cost.sort_values(['catalog', 'matches_checked']).reset_index(drop=True)

def get_label_index(results):
    '''Extracts the location of the true label in the list of predictions.'''
    
    label_idx = []
    for row in results.itertuples():
        preds = row.predictions.split(' ')
        test = index_mod(preds, row.id)
        label_idx.append(test)

    return label_idx

def index_mod(l, value):
    '''Same as list.index(), except returns len(list) on error.'''
    try:
        return l.index(value)
    except ValueError:
        return len(l)

def replace_bad_ids(results):
    '''Corrects some excel issues that arise when using MAC OS.'''
    is_bad = results.new_individual_id.str.contains('E+')
    bad_ids = results.loc[is_bad].new_individual_id
    bad_preds = results.loc[is_bad].predictions
    
    top_preds = [i.split(' ')[0] for i in bad_preds]

    df = pd.DataFrame({'bad_ids':bad_ids, 'top_preds':top_preds})

    first_few_ids = [i.replace('.', '')[:2] for i in bad_ids]
    first_few_preds = [i[:2] for i in top_preds]

    is_same = [i == j for i, j in zip(first_few_ids, first_few_preds)]

    key = df[is_same].drop_duplicates()

    lists = [key[k].tolist() for k in key]
    key_dict = dict(zip(lists[0], lists[1]))

    good_ids = results.new_individual_id.replace(key_dict)
    return good_ids 

def map_per_image(label, predictions):
    """Computes the precision score of one image."""    
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0

def classify_prediction(preds, label):
    """Classify predictions as (true, false) x (reject, accept)"""

    if label in preds:
        if label == 'new_individual':
            return 'TR'
        else:
            return 'TA'
    else:
        if 'new_individual' in preds:
            return 'FR'
        else:
            return 'FA'

def get_folder_specs(mapping, p_max=0.8, p_min=0.4):
    """get specs that are needed to build configs"""

    # number of images per id
    capture_count = (
        mapping.groupby('folder')['new_individual_id']
          .value_counts()
          .rename('img_per_id')
          .groupby('folder')
          .mean()
          .reset_index()
    )

    # number of ids in the training set (test set ids are ambiguous)
    train = mapping.loc[mapping.Usage == 'Train']
    id_count = (
        train.groupby('folder')
        .new_individual_id
        .nunique()
        .rename('id_count')
        .reset_index()
    )

    # add in species 
    species_maping = (
        mapping[['species', 'folder']]
          .drop_duplicates()
          .sort_values('species')
    )
    
    folder_specs = species_maping.merge(capture_count).merge(id_count)
    folder_specs = folder_specs.rename(columns={'folder':'catalog'})
    
    folder_specs['p'] = scale_p(folder_specs.img_per_id, p_max, p_min)
    folder_specs['N'] = (folder_specs['id_count'] / folder_specs['p']).astype(int)

    # optionally drop metadata 
    folder_specs.drop(columns=['id_count', 'img_per_id'], inplace=True)

    return folder_specs

def get_catalog_ids(species_list):
    '''change folder names to [beluga_0, beluga_1, ..., killer_whale_0, ...] '''
    values, counts = np.unique(species_list, return_counts=True)
    
    tmp = []
    for value, count in zip(values, counts):
        l = [f'{value}-{i}' for i in range(count)]
        tmp.append(l)
        
    catalog_ids = [item for sublist in tmp for item in sublist]
    
    return catalog_ids

def scale_p(mean, p_max, p_min):
    '''Scale the mean capture rate to the desired range.'''
    scaled = ((p_max - p_min) * (mean - min(mean)) / 
            (max(mean) - min(mean))) + p_min
    
    return scaled

if __name__ == '__main__':
    main()