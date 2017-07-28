import numpy as np

import pandas as pd

def make_clf_report(f_string,anno_string, unfair_df = False ):
    ''' f sting is the predictions, anno'''
    print("Classification report for predictions "+f_string)
    try:
        pred_df = pd.read_csv(f_string).iloc[:,1:]
    except:
        pred_df = pd.read_excel(f_string).iloc[:,1:]
    pred_df = add_mcode_tid_col(pred_df)

    try:
        anno_df = pd.read_csv(anno_string).iloc[:,1:]
    except:
        anno_df = pd.read_excel(anno_string).iloc[:,1:]
    anno_df  = add_mcode_tid_col(anno_df)
    if unfair_df:
        anno_unfair_df = add_mcode_tid_col(unfair_df)
        unfair_preds, fair_preds = compare_dfs(pred_df, anno_unfair_df)

    else:
        fair_preds = pred_df
    tp_df, fp_df = compare_dfs(fair_preds, anno_df)

    #print('TP:',tp_df.shape[0],anno_df.shape,'FP:',fp_df.shape[0], fair_preds.shape[0], tp_df.shape[0]+fp_df.shape[0])
    try:
        s_annos = anno_df['seizure duration'] # instead of anno_df.seizure duration.sum()
    except:
        s_annos = anno_df['duration'] # instead of anno_df.seizure duration.sum()

    print(tp_df.overlap.sum(),s_annos.sum(), (tp_df.overlap.sum()/s_annos.sum())*100,'%')

    anno_in_preds, anno_not_in_preds = compare_dfs(anno_df, tp_df) # anno_in_tp
    #print(anno_in_preds.shape, anno_df.shape)
    TPs = anno_in_preds.shape[0]
    FPs = fp_df.shape[0]
    total_Ps  = anno_df.shape[0]
    FNs = anno_not_in_preds.shape[0]
    precision = TPs/(TPs + FPs)
    recall    = TPs/(TPs+FNs)
    f1 = 2*((precision * recall) / (precision + recall))

    # Make table
    headers = ["TP","FP","FN ","precision", "recall", "f1-score",]
    fmt = '%% %ds' % 6  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'
    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    for i, label in enumerate(['ictal']):
        values = ['ictal']
        for v in (TPs,FPs,FNs, precision, recall, f1):
            values += ["{0:0.{1}f}".format(v, 3)]

        report += fmt % tuple(values)

    report += '\n'
    print(report)


def add_mcode_tid_col(df):
    df['mcode_tid'] = df.filename.str.slice(0,11)+'_'+df.transmitter.astype(str)
    return df

def check_overlap(series1,series2):
    ''' series should both have start and end attrs'''
    start_a, end_a = float(series1.start), float(series1.end)
    start_b, end_b = float(series2.start), float(series2.end)
    overlap_bool = (start_a <= end_b) and (end_a>=start_b)
    #print(overlap_bool,': ',start_a, '-',end_a,',', start_b,'-', end_b)
    return overlap_bool

def calculate_overlap(series1,series2):
    ''' series should both have start and end attrs
    http://baodad.blogspot.co.uk/2014/06/date-range-overlap.html
    '''
    a, b = float(series1.start), float(series1.end)
    c, d = float(series2.start), float(series2.end)
    b_a = b-a
    b_c = b-c
    d_c = d-c
    d_a = d-a
    overlap = min([b_a,b_c,d_c,d_a])
    #print(overlap,a,b,c,d)
    return overlap
# not quite there - need to make sure every seziure in annot is there, mult seizures when actuall one can mean could skip..?
def compare_dfs(prediction_df, annotation_df):
    ''' You should add a overlap calculation step to this...'''
    #annotations_found_df = annotation_df[['mcode_tid', 'seizure duration']]
    preds_in_anno    = pd.DataFrame(columns = prediction_df.columns)
    preds_notin_anno = pd.DataFrame(columns = prediction_df.columns)
    for _,prediction in prediction_df.iterrows():
        # check if hours and tid or prediciton in annotation df
        overlap_bool = 0
        if prediction.mcode_tid in annotation_df.mcode_tid.unique():
            # now filter annotation df for same hour and tid annotation as the prediction,
            # this should normally just be one row, but if >1 seizures in that hour will be more
            revevant_annotations_df = annotation_df[annotation_df.mcode_tid.isin([prediction.mcode_tid])]
            t_overlap = 0 # storing the overlap time
            for _,annotation in revevant_annotations_df.iterrows():
                row_overlap =  check_overlap(prediction,annotation) # in the case that two seizures, want to add...
                overlap_bool += row_overlap
                if row_overlap:# is this robust to two seizures?
                    t_overlap += calculate_overlap(prediction,annotation)

        if overlap_bool>0:
            prediction['overlap'] = t_overlap
            preds_in_anno = preds_in_anno.append(prediction)
        else:
            preds_notin_anno = preds_notin_anno.append(prediction)
    return preds_in_anno, preds_notin_anno#, annotations_found_df
