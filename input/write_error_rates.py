import pandas as pd
import numpy as np
import argparse

def parse():
    parser = argparse.ArgumentParser(description="Writing error rates to csv")
    parser.add_argument("--scenario", default="semi")
    return parser.parse_args()

def main():
    
    args = parse()

    SCENARIO = args.scenario
    
    SEMI_FALSE_ACCEPT_RATE = 0.02

    # directories
    base_dir = '/Users/philtpatton'
    onedrive = f'{base_dir}//OneDrive - hawaii.edu'

    # read in the mapping file 
    map_path = f'{base_dir}/datasets/hw/raw_data/full_file_mapping.csv'
    mapping = read_mapping(map_path)

    # add in the submission results
    sub_path = f'{onedrive}/projects/happy-whale/pds_submission_b7.csv'
    results = get_results(sub_path, mapping, SCENARIO)

    # calculate the rates of TP, FP, TN, FN
    error_rates = pd.crosstab(results.folder, results.pred_class, 
                              margins=False, normalize='index').reset_index()

    # combine rates and metadata
    folder_specs = get_folder_specs(mapping)
    error_rates = folder_specs.merge(error_rates)

    # add in ids for catalog
    catalog_ids = get_catalog_ids(error_rates.species.to_numpy())
    error_rates.insert(0, 'catalog_id', catalog_ids)

    # filter frasers
    error_rates = error_rates.loc[error_rates.species != 'frasiers_dolphin']

    # add zeros for test cases
    if args.scenario == 'test':
        error_rates['FP'] = 0
        error_rates['FN'] = 0
    
    # if args.scenario == 'semi5':
    #     error_rates['FP'] = SEMI_FALSE_ACCEPT_RATE
    #     error_rates = norm_rates(error_rates)

    # export to csv 
    repo_dir =  '/Users/philtpatton/source/repos/autocapture/'
    path = f'{repo_dir}/input/{SCENARIO}-rates.csv'        
    error_rates.to_csv(path, index=False)

    # export catalog ids
    path = f'{repo_dir}/input/catalog_ids.npy' 
    np.save(path, error_rates.catalog_id.to_numpy(), allow_pickle=True)       

    return None

def read_mapping(map_path):

    # read in the full file mapping, correcting the known species errors
    mapping = pd.read_csv(map_path, dtype='string')
    mapping['species'].replace({"globis": "short_finned_pilot_whale",
                               "pilot_whale": "short_finned_pilot_whale",
                               "kiler_whale": "killer_whale",
                               "pantropic_spotted_dolphin": "spotted_dolphin",
                               "bottlenose_dolpin": "bottlenose_dolphin"}, 
                               inplace=True)

    # correct the bottlenose/spinner error
    is_under = mapping['folder'] == 'Botlenose-Dolphin-Underwater'
    mapping.loc[is_under, 'species'] = 'spinner_dolphin'
    
    return mapping

def get_results(sub_path, mapping, scenario):
    
    # left join the model predictions from the submission to the file mapping, 
    submission = pd.read_csv(sub_path, dtype='string')

    # merge the submission with the mapping 
    results = submission.merge(mapping, left_on='image', 
                               right_on='img_name_new')
    results['new_individual_id'] = replace_bad_ids(results)

    # add in the precision for each prediction
    preds = results.predictions.str.split(' ').to_list()
    labs = results.new_individual_id.to_list()
    results['precision'] = [map_per_image(l, p) for l, p in zip(labs, preds)]
        
    if 'semi' in scenario:
        pred_class = [class_pred_semi(p, l) for p, l in zip(preds, labs)]

    # classify predicitions as (true + false) x (positive + negative) 
    top_pred = [pred[0] for pred in preds]
    if scenario == 'fully' or scenario == 'test':
        pred_class = [class_pred_fully(p, l) for p, l in zip(top_pred, labs)]

    if scenario == 'test':
        pred_class = np.where(np.array(top_pred) == np.array(labs), 'TP', 'TN')
        
    results['pred_class'] = pred_class
    
    return results 

def replace_bad_ids(results):
    
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

def class_pred_fully(pred, label):
    """Classify predictions as TP, FP, TN, FN."""

    if pred == 'new_individual':
        if pred == label:
            pred_class = 'TN'
        else:
            pred_class = 'FN'
    else:
        if pred == label:
            pred_class = 'TP'
        else:
            pred_class = 'FP'
        
    return pred_class

def class_pred_semi(preds, label):
    """Classify predictions as TP, FP, TN, FN."""

    rng = np.random.default_rng()
    coin_flip = rng.binomial(1, 0.0)

    if label in preds:
        if coin_flip:
            return 'FP'
        else:
            if label == 'new_individual':
                return 'TN'
            else:
                return 'TP'
    else:
        if 'new_individual' in preds:
            return 'FN'
        else:
            return 'FP'

def get_folder_specs(mapping):

    train = mapping.loc[mapping.Usage == 'Train']
    is_test = (mapping.Usage == 'Private') | (mapping.Usage == 'Public')
    test = mapping.loc[is_test]

    train_specs = (
        train.groupby('folder')
          .agg({'new_individual_id':pd.Series.nunique,
                'img_name_new':pd.Series.count})
          .rename(columns={'new_individual_id':'id_count',
                           'img_name_new':'train_img_count'})
          .reset_index()
    )

    capture_count = (
        train.groupby('folder')['new_individual_id']
          .value_counts()
          .rename('train_img_per_id')
          .groupby('folder')
          .mean()
          .reset_index()
    )

    species_map = (
        mapping[['species', 'folder']]
          .drop_duplicates()
          .sort_values('species')
    )
    
    folder_specs = species_map.merge(train_specs).merge(capture_count)
    
    return folder_specs

def get_catalog_ids(species_list):
    
    # here is some ugly code to get [beluga_0, beluga_1, ..., killer_whale_0]
    values, counts = np.unique(species_list, return_counts=True)
    
    tmp = []
    for value, count in zip(values, counts):
        l = [f'{value}-{i}' for i in range(count)]
        tmp.append(l)
        
    catalog_ids = [item for sublist in tmp for item in sublist]
    
    return catalog_ids

def norm_rates(error_rates): 

    tp = error_rates['TP']
    tn = error_rates['TN']
    fp = error_rates['FP']
    fn = error_rates['FN']

    tp = tp / (tp + tn + fp + fn)
    tn = tn / (tp + tn + fp + fn)
    fp = fp / (tp + tn + fp + fn)
    fn = fn / (tp + tn + fp + fn)

    error_rates['TP'] = tp
    error_rates['TN'] = tn
    error_rates['FP'] = fp
    error_rates['FN'] = fn
    
    return error_rates

if __name__ == '__main__':
    main()